"""SegmentTracker — mask cache and single write path for one segment.

Design
------
* ``_mask``  — live binary uint8 3-D array, always the authoritative state.
               All reads and writes go through this object.
* History is **not** stored here — it is owned by the widget's ``_history``
  list so there is one structure covering both current state and undo records.

Single write path
    ``write_slice()`` is the only function that mutates ``_mask`` and pushes to
    Slicer.  It computes ``delta = new − current``, crops it to the bounding
    box of changed pixels, updates ``_mask`` in-place, pushes to Slicer, and
    returns the ``MaskChange`` record.  Callers store that record in
    ``_history`` if they want it to be undoable.

Undo path
    ``reverse_delta(change)`` applies ``−delta`` to the bounding-box
    sub-region of ``_mask`` and pushes to Slicer — O(changed pixels).

Memory efficiency
    Only the bounding-box crop of each delta is stored per ``MaskChange``.
    A 10×10 stroke on a 512×512 slice stores 100 int16 values (200 B)
    rather than 512×512 (~500 KB).
"""

from collections import namedtuple
import logging
import numpy as np
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
            Positive values mark additions, negative values mark subtractions.
r_min     : int — top row of the bounding box in the slice.
c_min     : int — left column of the bounding box in the slice.
axis      : int — slice axis (0 axial / 1 coronal / 2 sagittal).
slice_idx : int — slice index along the axis.
source    : str — what produced this change ('brush' | 'prompt' | 'expand').
"""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SegmentTracker:
    """Mask cache and single write-back gate for one segment's 3-D labelmap.

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

        self._mask: np.ndarray | None = None   # lazy-loaded on first access

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.seg_node, self.segment_id, self.volume_node
        )
        self._mask = raw.copy()
        log.debug('[Tracker] loaded mask from Slicer key=%s', self._key)

    def _push_to_slicer(self):
        """Write the current _mask to Slicer and force closed-surface refresh."""
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self._mask, self.seg_node, self.segment_id, self.volume_node
        )
        self.seg_node.GetSegmentation().CreateRepresentation("Closed surface")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self):
        """Drop cached mask on next access.

        Call when an external tool has modified the segment outside our write
        path.  History (owned by the widget) is unaffected; callers that need
        to clear it should do so separately.
        """
        self._mask = None
        log.debug('[Tracker] invalidated key=%s', self._key)

    def sync(self):
        """Drop the cached mask so the next access reloads from Slicer.

        Preserves nothing — use before capturing a before-state so the
        snapshot reflects the latest MRML-committed data.
        """
        self._mask = None
        log.debug('[Tracker] sync (cache dropped) key=%s', self._key)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_mask(self) -> np.ndarray:
        """Return the live 3-D mask, loading from Slicer on first access."""
        if self._mask is None:
            self._load()
        return self._mask

    def get_slice(self, axis: int, idx: int) -> np.ndarray:
        """Return a view into the current mask at *axis / idx*."""
        return get_slice_from_volume(self.get_mask(), axis, idx)

    def snapshot(self) -> np.ndarray:
        """Deep copy of the full 3-D mask — used as the session base."""
        return self.get_mask().copy()

    # ------------------------------------------------------------------
    # Write API  (single path for ALL mask mutations)
    # ------------------------------------------------------------------

    def write_slice(self, axis: int, idx: int,
                    new_data: np.ndarray,
                    source: str = 'unknown') -> 'MaskChange | None':
        """Apply *new_data* to the tracked slice.

        Computes ``delta = int16(new_data) − int16(current_slice)``, crops it
        to the bounding box of changed pixels, updates ``_mask`` in-place,
        pushes the full mask to Slicer, and returns the ``MaskChange`` record.

        Returns ``None`` when *new_data* is identical to the current state
        (no-op — nothing is written and nothing is returned to store).
        """
        mask    = self.get_mask()
        current = get_slice_from_volume(mask, axis, idx)

        delta_full = new_data.astype(np.int16) - current.astype(np.int16)
        nz = np.argwhere(delta_full != 0)
        if len(nz) == 0:
            return None  # no actual change

        # Crop delta to bounding box of changed pixels
        r_min, c_min = nz.min(axis=0)
        r_max, c_max = nz.max(axis=0)
        delta_crop = delta_full[r_min:r_max + 1, c_min:c_max + 1].astype(np.int16)

        change = MaskChange(delta_crop, int(r_min), int(c_min), axis, idx, source)

        # Update _mask in-place and push to Slicer
        updated = (current.astype(np.int16) + delta_full > 0).astype(np.uint8)
        write_slice_to_volume(mask, updated, axis, idx)
        self._push_to_slicer()

        log.debug('[Tracker] wrote %s axis=%d slice=%d key=%s',
                  source, axis, idx, self._key)
        return change

    def reverse_delta(self, change: MaskChange):
        """Apply the inverse of *change* to ``_mask`` and push to Slicer.

        Subtracts the stored delta from the bounding-box sub-region —
        O(changed pixels), not O(full slice).
        """
        mask    = self.get_mask()
        current = get_slice_from_volume(mask, change.axis, change.slice_idx)

        r_end = change.r_min + change.delta.shape[0]
        c_end = change.c_min + change.delta.shape[1]

        sub = current[change.r_min:r_end, change.c_min:c_end].astype(np.int16)
        reversed_sub = (sub - change.delta > 0).astype(np.uint8)
        current[change.r_min:r_end, change.c_min:c_end] = reversed_sub

        self._push_to_slicer()
        log.debug('[Tracker] reversed %s change axis=%d slice=%d key=%s',
                  change.source, change.axis, change.slice_idx, self._key)
