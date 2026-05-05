"""SegmentTracker — zero-copy write path for one segment.

Design
------
No local copy of the 3-D mask is kept.  All reads and writes operate
directly on the segment's internal VTK image buffer obtained via
``GetBinaryLabelmapInternalRepresentation``.  ``vtk_to_numpy`` returns a
writable view into that buffer (zero-copy), so in-place numpy assignments
propagate directly to Slicer.  After each in-place edit, ``_notify()``
fires the VTK/MRML Modified events that trigger re-rendering.

This matches how Slicer's own C++ Paint/Erase effects work: they modify
the VTK buffer in-place at C++ speed rather than routing through the MRML
labelmap-volume pipeline (ExportSegmentsToLabelmapNode /
ImportLabelmapToSegmentationNode), which creates and destroys temporary
MRML nodes and deep-copies the full volume on every call.

Fallback
    When ``_vtk_view()`` returns ``(None, None)`` (non-zero extent origin,
    missing representation, or any VTK error), the affected method falls
    back to the MRML-pipeline helpers.  In practice this should not happen
    for segments initialised via SetReferenceImageGeometryParameterFromVolumeNode.

Single write path
    ``write_slice()`` is the only external entry-point that both modifies
    the VTK buffer and fires Modified.  Returns a ``MaskChange`` record.

Commit path (after Slicer's Paint/Erase effect)
    ``make_change(before, after)`` computes the delta without touching the
    buffer — the Paint effect already applied the stroke.

Undo / redo path
    ``reverse_delta`` / ``forward_delta`` apply ±delta to the bounding-box
    sub-region of the VTK buffer in-place — O(changed pixels), not O(volume).
"""

from collections import namedtuple
import logging
import numpy as np
import vtk.util.numpy_support as _vtk_ns
import slicer

from core.utils import get_slice_from_volume, write_slice_to_volume

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Change record
# ---------------------------------------------------------------------------

MaskChange = namedtuple(
    'MaskChange',
    ['delta', 'r_min', 'c_min', 'axis', 'slice_idx', 'source'],
)
"""
Fields
------
delta     : int16 ndarray of shape (h, w) — bounding-box crop of the change.
r_min     : int — top row of the bounding box in the slice.
c_min     : int — left column of the bounding box in the slice.
axis      : int — slice axis (0 axial / 1 coronal / 2 sagittal).
slice_idx : int — slice index along the axis.
source    : str — what produced this change ('brush' | 'erase' | 'prompt' | 'expand').
"""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SegmentTracker:
    """Zero-copy read/write gate for one segment's 3-D labelmap.

    Parameters
    ----------
    seg_node    : vtkMRMLSegmentationNode
    segment_id  : str
    volume_node : vtkMRMLScalarVolumeNode
    """

    def __init__(self, seg_node, segment_id, volume_node):
        self.seg_node    = seg_node
        self.segment_id  = segment_id
        self.volume_node = volume_node
        self._key        = (seg_node.GetID(), segment_id)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def matches(self, seg_node, segment_id, volume_node) -> bool:
        """True when the given triple identifies this tracker."""
        return (
            self._key == (seg_node.GetID(), segment_id)
            and self.volume_node is volume_node
        )

    # ------------------------------------------------------------------
    # VTK buffer access
    # ------------------------------------------------------------------

    def _vtk_view(self):
        """Return ``(numpy_view, vtkImageData)`` into the segment's VTK buffer.

        The view is zero-copy and writable: in-place numpy mutations modify
        Slicer's binary labelmap directly.  Call ``_notify(img)`` after edits.

        Returns ``(None, None)`` when the direct path is unavailable (missing
        representation, non-zero extent origin, degenerate dimensions, or
        unexpected exception).
        """
        try:
            lm = self.seg_node.GetBinaryLabelmapInternalRepresentation(self.segment_id)
            if lm is None:
                return None, None
            img = lm.GetImageData()
            ext = img.GetExtent()           # (xMin, xMax, yMin, yMax, zMin, zMax)
            if ext[0] != 0 or ext[2] != 0 or ext[4] != 0:
                return None, None           # non-zero origin — indices would be off
            dims = img.GetDimensions()      # (I, J, K) in VTK order
            if dims[0] <= 1 or dims[1] <= 1 or dims[2] <= 1:
                return None, None           # degenerate / uninitialised allocation
            flat = _vtk_ns.vtk_to_numpy(img.GetPointData().GetScalars())
            view = flat.reshape(dims[2], dims[1], dims[0])   # → (K, J, I)
            # Reject shared-layer buffers: Slicer sometimes packs multiple
            # segments into one image using per-segment label values > 1.
            # Our delta math assumes a binary (0/1) buffer.
            if flat.max() > 1:
                return None, None
            return view, img
        except Exception as exc:
            log.debug('[Tracker] _vtk_view failed: %s', exc)
            return None, None

    def ensure_own_layer(self) -> None:
        """Ensure the segment occupies its own binary labelmap layer.

        Slicer can collapse multiple segments into a shared VTK buffer
        (CollapseBinaryLabelmaps).  Call this before any direct VTK access so
        the buffer is guaranteed to be a private binary (0/1) array for this
        segment only.  No-op when the segment is already on its own layer.
        """
        try:
            self.seg_node.GetSegmentation().SeparateSegment(self.segment_id)
        except Exception as exc:
            log.debug('[Tracker] SeparateSegment failed (ignored): %s', exc)

    def _notify(self, img) -> None:
        """Fire Modified events so Slicer's render pipeline updates the display."""
        img.GetPointData().GetScalars().Modified()
        img.Modified()
        self.seg_node.GetSegmentation().GetSegment(self.segment_id).Modified()

    # ------------------------------------------------------------------
    # Slow-path fallbacks (MRML pipeline — used only when VTK path fails)
    # ------------------------------------------------------------------

    def _slow_read_slice(self, axis: int, idx: int) -> np.ndarray:
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.seg_node, self.segment_id, self.volume_node
        )
        return get_slice_from_volume(raw, axis, idx).copy()

    def _slow_write_slice(self, axis: int, idx: int, data: np.ndarray) -> None:
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.seg_node, self.segment_id, self.volume_node
        )
        vol = raw.copy()
        write_slice_to_volume(vol, data, axis, idx)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            vol, self.seg_node, self.segment_id, self.volume_node
        )

    # ------------------------------------------------------------------
    # Cache management (no-op — VTK view is always live)
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """No-op — the VTK buffer is always the live state, nothing to drop."""

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_slice(self, axis: int, idx: int) -> np.ndarray:
        """Return a 2-D copy of the current mask at *axis / idx*."""
        view, _ = self._vtk_view()
        if view is not None:
            return get_slice_from_volume(view, axis, idx).copy()
        return self._slow_read_slice(axis, idx)

    def get_mask(self) -> np.ndarray:
        """Return a full 3-D copy of the current mask.

        Prefer ``get_slice()`` for single-slice operations — this copies the
        entire volume and should only be used by model families that need the
        full image (e.g. SPX).
        """
        view, _ = self._vtk_view()
        if view is not None:
            return view.copy()
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.seg_node, self.segment_id, self.volume_node
        )
        return raw.copy()

    def snapshot(self) -> np.ndarray:
        """Deep copy of the full 3-D mask — used as the session base."""
        return self.get_mask()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_delta(delta_full: np.ndarray,
                    axis: int, idx: int, source: str) -> 'MaskChange | None':
        nz = np.argwhere(delta_full != 0)
        if len(nz) == 0:
            return None
        r_min, c_min = nz.min(axis=0)
        r_max, c_max = nz.max(axis=0)
        delta_crop = delta_full[r_min:r_max + 1, c_min:c_max + 1].astype(np.int16)
        return MaskChange(delta_crop, int(r_min), int(c_min), axis, idx, source)

    def make_change(self, axis: int, idx: int,
                    before_slice: np.ndarray, after_slice: np.ndarray,
                    source: str = 'brush') -> 'MaskChange | None':
        """Return the MaskChange for a stroke already applied by Slicer's effect.

        Does NOT touch the VTK buffer — the Paint/Erase effect committed the
        stroke.  Only the bounding-box delta is computed and returned for
        storage in the undo stack.
        """
        delta_full = after_slice.astype(np.int16) - before_slice.astype(np.int16)
        return self._crop_delta(delta_full, axis, idx, source)

    def write_slice(self, axis: int, idx: int,
                    new_data: np.ndarray,
                    source: str = 'unknown') -> 'MaskChange | None':
        """Apply *new_data* to the tracked slice.

        Modifies the VTK buffer in-place, fires Modified(), and returns
        the MaskChange record.  Returns None when there is no net change.
        """
        view, img = self._vtk_view()
        if view is not None:
            current    = get_slice_from_volume(view, axis, idx)
            delta_full = new_data.astype(np.int16) - current.astype(np.int16)
            change     = self._crop_delta(delta_full, axis, idx, source)
            if change is None:
                return None
            updated = (current.astype(np.int16) + delta_full > 0).astype(np.uint8)
            write_slice_to_volume(view, updated, axis, idx)
            self._notify(img)
            log.debug('[Tracker] wrote %s axis=%d slice=%d key=%s',
                      source, axis, idx, self._key)
            return change

        # Slow fallback
        before = self._slow_read_slice(axis, idx)
        delta_full = new_data.astype(np.int16) - before.astype(np.int16)
        change = self._crop_delta(delta_full, axis, idx, source)
        if change is None:
            return None
        updated = (before.astype(np.int16) + delta_full > 0).astype(np.uint8)
        self._slow_write_slice(axis, idx, updated)
        log.debug('[Tracker] wrote(slow) %s axis=%d slice=%d key=%s',
                  source, axis, idx, self._key)
        return change

    def reverse_delta(self, change: MaskChange) -> None:
        """Apply the inverse of *change* to the VTK buffer (undo path)."""
        view, img = self._vtk_view()
        if view is not None:
            current = get_slice_from_volume(view, change.axis, change.slice_idx)
            r_end   = change.r_min + change.delta.shape[0]
            c_end   = change.c_min + change.delta.shape[1]
            sub     = current[change.r_min:r_end, change.c_min:c_end].astype(np.int16)
            current[change.r_min:r_end, change.c_min:c_end] = (
                (sub - change.delta > 0).astype(np.uint8)
            )
            self._notify(img)
            log.debug('[Tracker] reversed %s axis=%d slice=%d key=%s',
                      change.source, change.axis, change.slice_idx, self._key)
            return

        # Slow fallback
        before = self._slow_read_slice(change.axis, change.slice_idx)
        r_end  = change.r_min + change.delta.shape[0]
        c_end  = change.c_min + change.delta.shape[1]
        sub    = before[change.r_min:r_end, change.c_min:c_end].astype(np.int16)
        before[change.r_min:r_end, change.c_min:c_end] = (
            (sub - change.delta > 0).astype(np.uint8)
        )
        self._slow_write_slice(change.axis, change.slice_idx, before)
        log.debug('[Tracker] reversed(slow) %s axis=%d slice=%d key=%s',
                  change.source, change.axis, change.slice_idx, self._key)

    def forward_delta(self, change: MaskChange) -> None:
        """Re-apply *change* to the VTK buffer (redo path)."""
        view, img = self._vtk_view()
        if view is not None:
            current = get_slice_from_volume(view, change.axis, change.slice_idx)
            r_end   = change.r_min + change.delta.shape[0]
            c_end   = change.c_min + change.delta.shape[1]
            sub     = current[change.r_min:r_end, change.c_min:c_end].astype(np.int16)
            current[change.r_min:r_end, change.c_min:c_end] = (
                (sub + change.delta > 0).astype(np.uint8)
            )
            self._notify(img)
            log.debug('[Tracker] re-applied %s axis=%d slice=%d key=%s',
                      change.source, change.axis, change.slice_idx, self._key)
            return

        # Slow fallback
        before = self._slow_read_slice(change.axis, change.slice_idx)
        r_end  = change.r_min + change.delta.shape[0]
        c_end  = change.c_min + change.delta.shape[1]
        sub    = before[change.r_min:r_end, change.c_min:c_end].astype(np.int16)
        before[change.r_min:r_end, change.c_min:c_end] = (
            (sub + change.delta > 0).astype(np.uint8)
        )
        self._slow_write_slice(change.axis, change.slice_idx, before)
        log.debug('[Tracker] re-applied(slow) %s axis=%d slice=%d key=%s',
                  change.source, change.axis, change.slice_idx, self._key)
