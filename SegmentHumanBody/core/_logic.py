"""Business logic for the SegmentHumanBody Slicer module.

Separated from SegmentHumanBody.py so the Widget file stays focused on UI.
All Slicer / VTK / Qt imports live here; core/ remains pure-Python.
"""

import logging
import vtk
import slicer
import numpy as np

# Set True to print per-phase timing inside apply_spx_labels_batch to the Python console.
SPX_DEBUG_TIMING = True
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from core.modelFamilies import SPXModelFamily
from core.utils import (
    call_if_exists,
    get_slice_from_volume,
    write_slice_to_volume,
    apply_window_level,
    spx_boundary_mask,
    select_spx_labels,
    POSITION_DEFINED,
    VIEW_TO_AXIS,
    AXIS_TO_IJK_COMPONENT,
    ras_to_ijk_2d,
)
from core._tracker import SegmentTracker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter-node reference names
# ---------------------------------------------------------------------------

POS_NODE   = 'positivePromptPointsNode'
NEG_NODE   = 'negativePromptPointsNode'
INPUT_VOLUME = 'InputVolume'
SEGMENTATION = 'Segmentation'

# Config shared by ensurePromptNodesExist and recreatePromptNodes.
# Each entry: ref_name → (RGB color, display label)
_PROMPT_NODE_CONFIGS = {
    POS_NODE: ([0, 1, 0], 'positive'),
    NEG_NODE: ([1, 0, 0], 'negative'),
}


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _node_records(node):
    """Return [(status, cp_id, position), …] for all control points in *node*.

    Produces the format expected by collect_confirmed_points /
    collect_preview_points.  Returns an empty list when *node* is None.
    """
    if not node:
        return []
    return [
        (node.GetNthControlPointPositionStatus(i),
         node.GetNthControlPointID(i),
         node.GetNthControlPointPosition(i))
        for i in range(node.GetNumberOfControlPoints())
    ]


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Numpy cache for the active segment; routes all reads/writes through
        # a single object so Slicer stays in sync.  Replaced lazily by
        # _get_tracker() when the segment identity changes.
        self._tracker: SegmentTracker | None = None

        # Confirmed W/L values (set via "Apply Window/Level").  When set, each
        # slice is normalized to [0, 255] before reaching the model.  Volume
        # data is never modified.
        self._wl_window = None
        self._wl_level  = None

    def setDefaultParameters(self, parameterNode):
        pass

    # -------------------------
    # Cache management
    # -------------------------

    def _get_tracker(self, segNode, segmentID, volumeNode) -> SegmentTracker:
        """Return the active ``SegmentTracker``, creating one when necessary.

        A new tracker is created when the segment identity changes (node ID or
        segment ID).  The tracker is intentionally preserved across
        ``reset_render_state()`` calls since it holds the authoritative 3D mask.
        The tracker lazily loads the numpy mask from Slicer on first access.

        Also ensures the closed surface representation exists so that writes
        to the binary labelmap are immediately reflected in the 3D view.
        ``CreateClosedSurfaceRepresentation`` is a no-op if the representation
        already exists, so this check is cheap on subsequent calls.
        """
        if self._tracker is None or not self._tracker.matches(segNode, segmentID, volumeNode):
            self._tracker = SegmentTracker(segNode, segmentID, volumeNode)
            segNode.CreateClosedSurfaceRepresentation()
        return self._tracker

    def get_working_mask(self, segNode, segmentID, volumeNode):
        """Return the committed 3-D segment mask (public shim over the tracker).

        Callers outside ``_logic.py`` (e.g. ``_captureSegmentationState`` in
        the widget) use this method.  For read-only access prefer the returned
        array to be treated as immutable; writes must go through
        ``restore_slice()`` or the tracker's ``write_slice()`` directly.
        """
        return self._get_tracker(segNode, segmentID, volumeNode).get_mask()

    def capture_current_slice(self, widget):
        """Return the current slice as a before-state snapshot for stroke undo.

        ``tracker.get_slice()`` reads directly from the segment's VTK buffer
        (zero-copy view → 2-D slice copy) — O(H×W), not O(H×W×D).

        Returns
        -------
        (axis, idx, slice_copy) — coordinates and a numpy copy of the slice,
        or (None, None, None) when the required selectors are empty.
        """
        vol, seg, seg_id = self._get_context(widget)
        if not vol or not seg or not seg_id:
            return None, None, None
        axis, idx = self.getAxisAndSlice(widget, vol)
        tracker = self._get_tracker(seg, seg_id, vol)
        before_slice = tracker.get_slice(axis, idx)   # already returns a copy
        return axis, idx, before_slice

    def commit_stroke(self, widget, axis, idx, before_slice, source='brush') -> 'MaskChange | None':
        """Record a brush stroke as a tracked delta.

        Reads the after-state from the segment's VTK buffer (zero-copy via
        tracker._vtk_view), computes delta = after − before, and returns the
        MaskChange.  Does NOT write back to Slicer — the Paint/Erase effect
        already applied the stroke.

        Falls back to arrayFromSegmentBinaryLabelmap if the VTK path is
        unavailable.

        Returns None when the stroke produced no net change.
        """
        vol, seg, seg_id = self._get_context(widget)
        if not vol or not seg or not seg_id:
            return None
        tracker = self._get_tracker(seg, seg_id, vol)

        # Fast path: tracker._vtk_view() is a zero-copy numpy view into the
        # VTK buffer — no temp MRML nodes, no full-volume copies.
        view, _ = tracker._vtk_view()
        if view is not None:
            after_slice = get_slice_from_volume(view, axis, idx).copy()
            return tracker.make_change(axis, idx, before_slice, after_slice, source)

        # Slow fallback — ExportSegmentsToLabelmapNode through a temp MRML node.
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(seg, seg_id, vol)
        if raw is None:
            log.warning('[Logic] commit_stroke: labelmap read returned None — stroke lost')
            return None
        after_slice = get_slice_from_volume(raw, axis, idx).copy()
        return tracker.make_change(axis, idx, before_slice, after_slice, source)

    def warmup_tracker(self, widget) -> None:
        """Prepare the tracker for the active segment before the first stroke.

        Called during brush/erase activation.  Ensures the segment occupies
        its own binary labelmap layer (SeparateSegment) and that
        GetBinaryLabelmapInternalRepresentation has been called at least once.
        Any slow one-time work (representation conversion, layer separation,
        buffer reallocation) happens here — at button-click time — rather than
        during the first _on_stroke_start, which would block the UI thread and
        cause the first stroke to appear as a straight line.
        """
        vol, seg, seg_id = self._get_context(widget)
        if not (vol and seg and seg_id):
            return
        tracker = self._get_tracker(seg, seg_id, vol)
        tracker.ensure_own_layer()          # separate from any shared layer
        axis, idx = self.getAxisAndSlice(widget, vol)
        tracker.get_slice(axis, idx)        # prime _vtk_view() / fall back once

    def _apply_delta(self, widget, change, method):
        if change is None:
            return
        vol, seg, seg_id = self._get_context(widget)
        if not vol or not seg or not seg_id:
            return
        getattr(self._get_tracker(seg, seg_id, vol), method)(change)

    def reverse_change(self, widget, change) -> None:
        """Apply the inverse of *change* to the tracker and push to Slicer."""
        self._apply_delta(widget, change, 'reverse_delta')

    def forward_change(self, widget, change) -> None:
        """Re-apply *change* to the tracker and push to Slicer (redo path)."""
        self._apply_delta(widget, change, 'forward_delta')

    # -------------------------
    # Window / Level
    # -------------------------

    def set_window_level(self, window, level):
        """Confirm W/L values for model inference.
        Subsequent calls to onRender / on_expand will normalize each slice
        to [0, 255] using these values before passing it to the model.
        Call with (None, None) to revert to raw values.
        """
        self._wl_window = window
        self._wl_level  = level

    def _apply_wl_to_slice(self, img):
        """Delegate to ``apply_window_level`` using the confirmed W/L values.
        Returns the original array unchanged when no W/L has been confirmed.
        The source volume data is never modified.
        """
        return apply_window_level(img, self._wl_window, self._wl_level)

    # -------------------------
    # Prompt Nodes
    # -------------------------

    def setVolumeAndSegmentation(self, parameterNode, volumeNode, segmentationNode):
        if volumeNode:
            parameterNode.SetNodeReferenceID(INPUT_VOLUME, volumeNode.GetID())
        if segmentationNode:
            parameterNode.SetNodeReferenceID(SEGMENTATION, segmentationNode.GetID())

    def getVolumeAndSegmentation(self, parameterNode):
        return (
            parameterNode.GetNodeReference(INPUT_VOLUME),
            parameterNode.GetNodeReference(SEGMENTATION),
        )

    # -------------------------
    # Context helper
    # -------------------------

    def _get_context(self, widget):
        """Extract ``(volumeNode, segNode, segmentID)`` from the widget's selectors.

        Centralises repeated ``widget.ui.X.currentNode()`` calls so that
        changing the selector structure only requires updating this one method.
        """
        return (
            widget.ui.sourceVolumeSelector.currentNode(),
            widget.ui.segmentSelector.currentNode(),
            widget.ui.segmentSelector.currentSegmentID(),
        )

    # -------------------------
    # Prompt Nodes
    # -------------------------

    def _make_prompt_node(self, parameterNode, ref_name, color, label):
        """Remove any existing node for *ref_name*, create a fresh one, and
        register it on *parameterNode*.  Fresh nodes have a label counter of 0
        so the placement cursor is always labeled 'Positive 1' / 'Negative 1'.
        """
        old = parameterNode.GetNodeReference(ref_name)
        if old:
            log.debug("[PromptNode] Removing old node '%s' (%s)", label, old.GetID())
            slicer.mrmlScene.RemoveNode(old)
        node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', label)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetSelectedColor(*color)
        dn.SetColor(*color)
        dn.SetActiveColor(*color)
        node.SetHideFromEditors(True)
        parameterNode.SetNodeReferenceID(ref_name, node.GetID())
        return node

    def ensurePromptNodesExist(self, parameterNode):
        """Create prompt nodes for any reference slot that is currently empty."""
        for ref_name, (color, label) in _PROMPT_NODE_CONFIGS.items():
            if not parameterNode.GetNodeReference(ref_name):
                self._make_prompt_node(parameterNode, ref_name, color, label)

    def recreatePromptNodes(self, parameterNode):
        """Replace both prompt nodes with brand-new ones (counter reset to 0)."""
        for ref_name, (color, label) in _PROMPT_NODE_CONFIGS.items():
            self._make_prompt_node(parameterNode, ref_name, color, label)

    def recreate_prompt_node(self, parameterNode, is_negative: bool):
        """Replace one prompt node with a fresh one (counter reset to 0).

        Called after undo empties a node so the next placement cursor shows
        the correct label ('Positive 1' / 'Negative 1') rather than N+1.
        Returns the newly created node.
        """
        ref_name = NEG_NODE if is_negative else POS_NODE
        color, label = _PROMPT_NODE_CONFIGS[ref_name]
        return self._make_prompt_node(parameterNode, ref_name, color, label)

    def setPromptNodes(self, parameterNode, posNode, negNode):
        parameterNode.SetNodeReferenceID(
            POS_NODE, posNode.GetID() if posNode else None
        )
        parameterNode.SetNodeReferenceID(
            NEG_NODE, negNode.GetID() if negNode else None
        )

    def getPromptNodes(self, parameterNode):
        return (
            parameterNode.GetNodeReference(POS_NODE),
            parameterNode.GetNodeReference(NEG_NODE),
        )

    def _ensure_seg_and_segment(self, widget, volumeNode):
        """Guarantee a segmentation node and at least one segment exist.

        Creates them if absent and updates the UI selectors (with signals
        blocked so no downstream cascades fire).  Returns (segNode, segmentID).
        Safe to call multiple times — a no-op when everything already exists.
        """
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not segNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            segNode.CreateClosedSurfaceRepresentation()
            self.setVolumeAndSegmentation(widget._parameterNode, volumeNode, segNode)
            widget.ui.segmentationNodeSelector.blockSignals(True)
            widget.ui.segmentationNodeSelector.setCurrentNode(segNode)
            widget.ui.segmentationNodeSelector.blockSignals(False)
            widget.ui.segmentSelector.blockSignals(True)
            widget.ui.segmentSelector.setCurrentNode(segNode)
            widget.ui.segmentSelector.blockSignals(False)

        if not segmentID:
            # Block the segmentSelector signal AND set the _creating_segment flag
            # before AddEmptySegment fires its VTK event.  qMRMLSegmentSelectorWidget
            # reacts to that VTK event synchronously and would emit currentSegmentChanged,
            # triggering onSegmentChanged → clearPrompts() in the middle of a render.
            widget.ctrl.creating_segment = True
            widget.ui.segmentSelector.blockSignals(True)
            try:
                segmentID = segNode.GetSegmentation().AddEmptySegment("Segment_1")
                widget.ui.segmentSelector.setCurrentSegmentID(segmentID)
                widget.ui.addSegmentButton.setEnabled(True)
            finally:
                widget.ui.segmentSelector.blockSignals(False)
                widget.ctrl.creating_segment = False
            # Pre-acknowledge the new segment so the deferred currentSegmentChanged
            # emitted by qMRMLSegmentSelectorWidget's internal QTimer after
            # blockSignals(False) is treated as a duplicate by onSegmentChanged
            # and does not trigger clearPrompts() or wipe _history.
            if segmentID:
                widget._acknowledged_segment_id = segmentID

        return segNode, segmentID

    # -------------------------
    # Point commit
    # -------------------------

    def commit_point(self, widget, node, cp_id) -> 'MaskChange | None':
        """Compute and write the superpixel selection for one control point.

        Runs the SPX model on the current slice (reusing its label cache when
        available), finds the superpixel at the point's 2-D position, and writes
        current | label_pixels (positive) or current & ~label_pixels (negative)
        through SegmentTracker.write_slice() — the same single write path used
        by brush and erase strokes.
        """
        vol = widget.ui.sourceVolumeSelector.currentNode()
        if not vol:
            return None

        seg, seg_id = (widget.ui.segmentSelector.currentNode(),
                       widget.ui.segmentSelector.currentSegmentID())
        if not seg or not seg_id:
            self._ensure_seg_and_segment(widget, vol)
            seg  = widget.ui.segmentSelector.currentNode()
            seg_id = widget.ui.segmentSelector.currentSegmentID()
        if not seg or not seg_id:
            return None

        modelFamily = widget.modelFamily
        if not modelFamily or not modelFamily.model:
            return None

        posNode, negNode = self.getPromptNodes(widget._parameterNode)
        is_negative = (node is negNode)

        idx_in_node = node.GetControlPointIndexByID(cp_id)
        if idx_in_node < 0:
            return None

        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(idx_in_node, ras)

        axis, slice_idx = self.getAxisAndSlice(widget, vol)

        # Convert RAS → 2-D slice coordinates; off-slice points are filtered out.
        scrib_key = 'negative' if is_negative else 'positive'
        scrib = {'positive': [], 'negative': []}
        scrib[scrib_key] = [ras]
        pts_2d = self.ras_to_ijk(vol, scrib, axis, slice_index=slice_idx)[scrib_key]
        if not pts_2d:
            return None  # point was placed on a different slice

        px, py = int(round(pts_2d[0][0])), int(round(pts_2d[0][1]))

        volumeArray = slicer.util.arrayFromVolume(vol)
        if volumeArray is None:
            return None
        img = get_slice_from_volume(volumeArray, axis, slice_idx)
        img = self._apply_wl_to_slice(img)

        params = widget.getUserParameters() or {}
        labels = call_if_exists(modelFamily, 'on_expand', img=img, **params)
        if labels is None:
            return None

        if not (0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]):
            return None

        spx_pixels = labels == labels[py, px]

        tracker = self._get_tracker(seg, seg_id, vol)
        current  = tracker.get_slice(axis, slice_idx).astype(bool)

        if is_negative:
            new_data = (current & ~spx_pixels).astype(np.uint8)
            source   = 'neg_prompt'
        else:
            new_data = (current | spx_pixels).astype(np.uint8)
            source   = 'prompt'

        return tracker.write_slice(axis, slice_idx, new_data, source=source)

    # -------------------------
    # -------------------------
    # Model actions
    # -------------------------

    def on_confirm_model(self, widget):
        if not widget.modelFamily:
            return
        widget.modelFamily.confirm_model()

    def on_expand(self, widget) -> 'MaskChange | None':
        """Run the SPX expansion and return the MaskChange for the widget to store.

        Returns ``None`` when a pre-condition fails (no model, no volume, etc.)
        or when the expansion produced no net change.  The widget is responsible
        for appending the returned change to its ``_history`` list.
        """
        modelFamily = widget.modelFamily

        if not modelFamily:
            slicer.util.warningDisplay("Please select a model first.")
            return None

        if not getattr(modelFamily, "model", None):
            slicer.util.warningDisplay("Please click 'Confirm Model Selection' before running.")
            return None

        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not volumeNode:
            slicer.util.warningDisplay("Please select a source volume.")
            return None

        if not segNode or not segmentID:
            slicer.util.warningDisplay("Please select a segmentation and segment.")
            return None

        params = widget.getUserParameters()
        if params is None:
            slicer.util.warningDisplay("Invalid model parameters.")
            return None

        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        if volumeArray is None:
            slicer.util.warningDisplay("Volume has no image data.")
            return None
        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)

        # Collect confirmed (PositionDefined) negative prompt points only.
        _, negNode = self.getPromptNodes(widget._parameterNode)
        neg_confirmed = [pos for (status, _, pos) in _node_records(negNode)
                         if status == POSITION_DEFINED]
        neg_ijk = self.ras_to_ijk(
            volumeNode, {"positive": [], "negative": neg_confirmed}, axis,
            slice_index=sliceIndex,
        )["negative"]

        # Delegate to the family so the correct algorithm and user params are
        # used, and the SPX label cache is consulted before recomputing.
        labels = call_if_exists(modelFamily, 'on_expand',
                                volume_node=volumeNode, axis=axis,
                                slice_idx=sliceIndex, img=img, **params)

        if labels is None:
            slicer.util.warningDisplay("This model does not support propagation.")
            return None

        return self.expandSegWithSPX(
            segNode, segmentID, volumeNode, labels, axis, sliceIndex,
            neg_points=neg_ijk,
        )

    # -------------------------
    # Coordinate helpers
    # -------------------------

    def ras_to_ijk(self, volumeNode, scrib, axis, slice_index=None):
        # Extract the 4×4 RAS-to-IJK matrix from VTK into numpy, then delegate
        # the batch point conversion to the Slicer-agnostic ras_to_ijk_2d helper.
        # slice_index filters out points placed on a different slice so they
        # cannot drive model results on the current slice.
        vtk_mat = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(vtk_mat)
        mat = np.array([[vtk_mat.GetElement(r, c) for c in range(4)]
                        for r in range(4)])
        return {
            "positive": ras_to_ijk_2d(mat, scrib["positive"], axis, slice_index),
            "negative": ras_to_ijk_2d(mat, scrib["negative"], axis, slice_index),
        }

    def getAxisAndSlice(self, widget, volumeNode=None):
        viewName = widget.currentViewName
        axis = VIEW_TO_AXIS.get(viewName, 0)

        lm = slicer.app.layoutManager()
        sliceWidget = lm.sliceWidget(viewName)

        if volumeNode is not None:
            # Convert the slice plane's RAS origin to the volume's IJK space so
            # that the index is correct regardless of the volume's spacing/origin.
            sliceNode = sliceWidget.mrmlSliceNode()
            sliceToRAS = sliceNode.GetSliceToRAS()
            ras = [sliceToRAS.GetElement(r, 3) for r in range(3)]
            rasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(rasToIjk)
            ijk = rasToIjk.MultiplyPoint(ras + [1])
            sliceIndex = int(round(ijk[AXIS_TO_IJK_COMPONENT[axis]]))
        else:
            logic = sliceWidget.sliceLogic()
            sliceIndex = logic.GetSliceIndexFromOffset(logic.GetSliceOffset()) - 1

        return axis, sliceIndex

    # -------------------------
    # SPX expansion
    # -------------------------

    def expandSegWithSPX(self, segNode, segmentID, volumeNode,
                         labels, axis, sliceIndex,
                         neg_points=None) -> 'MaskChange | None':
        """Apply SPX label selection and return the MaskChange (or None if no-op)."""
        tracker = self._get_tracker(segNode, segmentID, volumeNode)
        expanded = select_spx_labels(
            labels,
            tracker.get_slice(axis, sliceIndex),
            neg_points=neg_points,
        )
        return tracker.write_slice(axis, sliceIndex, expanded, source='expand')


# ---------------------------------------------------------------------------
# Module-level SPX helpers — imported directly by core/_input.py
# ---------------------------------------------------------------------------

def _build_slice_modifier_labelmap(vol, axis, slice_idx, mask2d,
                                   row_offset=0, col_offset=0,
                                   ijk_to_ras_mat=None, modifier=None):
    """Build a sub-window vtkOrientedImageData for modifySelectedSegmentByLabelmap.

    mask2d is the (cropped) 2-D uint8 mask to write.  row_offset / col_offset are the
    top-left corner of that crop within the full slice in slice-local 2-D coordinates:
      axis 0: row=J, col=I  →  origin_ijk = (col_offset, row_offset, slice_idx)
      axis 1: row=K, col=I  →  origin_ijk = (col_offset, slice_idx,  row_offset)
      axis 2: row=K, col=J  →  origin_ijk = (slice_idx,  col_offset, row_offset)

    ijk_to_ras_mat: optional pre-fetched vtkMatrix4x4 from vol.GetIJKToRASMatrix().
                    When provided, the per-call GetIJKToRASMatrix() read is skipped.
    modifier:       optional vtkOrientedImageData to reuse.  When its total voxel
                    count matches slim_dims, scalars are updated in-place (no new
                    VTK array, no deep copy).  Caller should cache the returned
                    modifier and pass it on the next stroke.

    Callers that pass the full slice (fill_hole_2d) leave offsets at 0 — same behaviour.
    modifySelectedSegmentByLabelmap handles partial-extent modifiers via
    vtkOrientedImageDataResample::ModifyImage — only the overlap region is written.
    """
    from vtk.util.numpy_support import vtk_to_numpy
    m  = mask2d.astype(np.uint8)
    nr = m.shape[0]   # rows in cropped mask
    nc = m.shape[1]   # cols in cropped mask

    # numpy layout (nk,nj,ni) → ravel C-order → VTK expects i varies fastest, matching.
    if axis == 0:          # axial K: one K-slice → shape (1, nj_crop, ni_crop)
        arr        = m[np.newaxis, :, :]
        slim_dims  = (nc, nr, 1)
        origin_ijk = (float(col_offset), float(row_offset), float(slice_idx), 1.0)
    elif axis == 1:        # coronal J: one J-slice → shape (nk_crop, 1, ni_crop)
        arr        = m[:, np.newaxis, :]
        slim_dims  = (nc, 1, nr)
        origin_ijk = (float(col_offset), float(slice_idx), float(row_offset), 1.0)
    else:                  # sagittal I: one I-slice → shape (nk_crop, nj_crop, 1)
        arr        = m[:, :, np.newaxis]
        slim_dims  = (1, nc, nr)
        origin_ijk = (float(slice_idx), float(col_offset), float(row_offset), 1.0)

    # Use cached matrix when available; otherwise read from volume node.
    if ijk_to_ras_mat is None:
        ijk_to_ras_mat = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(ijk_to_ras_mat)

    origin_ras = [0.0, 0.0, 0.0, 0.0]
    ijk_to_ras_mat.MultiplyPoint(origin_ijk, origin_ras)
    slim_mat = vtk.vtkMatrix4x4()
    slim_mat.DeepCopy(ijk_to_ras_mat)
    slim_mat.SetElement(0, 3, origin_ras[0])
    slim_mat.SetElement(1, 3, origin_ras[1])
    slim_mat.SetElement(2, 3, origin_ras[2])

    # Reuse the modifier object when the total voxel count matches (most strokes).
    # Otherwise create or resize, then allocate.
    total = slim_dims[0] * slim_dims[1] * slim_dims[2]
    if modifier is None:
        modifier = slicer.vtkOrientedImageData()
    existing = modifier.GetPointData().GetScalars()
    if existing is None or existing.GetNumberOfTuples() != total:
        modifier.SetDimensions(slim_dims)
        modifier.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        existing = modifier.GetPointData().GetScalars()
    else:
        modifier.SetDimensions(slim_dims)  # update shape metadata; scalars stay

    # Write mask data directly into the allocated buffer — no new VTK array.
    np_scalars = vtk_to_numpy(existing)
    np_scalars[:] = arr.ravel()
    existing.Modified()

    modifier.SetGeometryFromImageToWorldMatrix(slim_mat)
    return modifier


def _read_segment_slice(seg, seg_id, vol, axis, slice_idx):
    """Read the current binary mask for one segment/slice as a 2-D bool array."""
    arr3d = slicer.util.arrayFromSegmentBinaryLabelmap(seg, seg_id, vol)
    if arr3d is None:
        dims = vol.GetImageData().GetDimensions()  # (ni, nj, nk)
        if axis == 0:
            return np.zeros((dims[2], dims[1]), dtype=bool)
        elif axis == 1:
            return np.zeros((dims[2], dims[0]), dtype=bool)
        else:
            return np.zeros((dims[1], dims[0]), dtype=bool)
    if axis == 0:
        return arr3d[slice_idx, :, :].astype(bool)
    elif axis == 1:
        return arr3d[:, slice_idx, :].astype(bool)
    else:
        return arr3d[:, :, slice_idx].astype(bool)


def _count_delta(delta):
    """Pixel count from modifySelectedSegmentByLabelmap (int) or 0 if unavailable."""
    return delta if isinstance(delta, int) else 0


def apply_spx_label(widget, label_id, axis, slice_idx, additive):
    """Fill or erase one superpixel (label_id) in the active segment.

    Calls widget.modelFamily.on_expand to get the cached label map, extracts
    the mask for label_id, and applies it via the Segment Editor effect API.
    All mask writes go through modifySelectedSegmentByLabelmap so Slicer's
    masking rules, observers, and undo snapshot are honoured automatically.
    """
    editor = widget.logic.get_segment_editor()
    if editor is None:
        return None
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    vol    = editor.sourceVolumeNode()
    if not (seg and seg_id and vol):
        return None

    vol_arr = slicer.util.arrayFromVolume(vol)
    img = get_slice_from_volume(vol_arr, axis, int(slice_idx))
    labels = widget.modelFamily.on_expand(
        volume_node=vol, axis=axis, slice_idx=int(slice_idx),
        img=img, **widget.getUserParameters())
    if labels is None:
        return None
    mask2d = (labels == label_id)
    if not mask2d.any():
        return None

    modifier = _build_slice_modifier_labelmap(vol, axis, int(slice_idx), mask2d)
    from slicer import qSlicerSegmentEditorAbstractEffect as _Eff
    mode = _Eff.ModificationModeAdd if additive else _Eff.ModificationModeRemove
    effect = editor.effectByName('Paint')
    if effect is None:
        return None
    delta = effect.modifySelectedSegmentByLabelmap(modifier, mode)

    if widget._recorder.is_active:
        widget._recorder.record_spx_fill(
            additive=additive,
            label_id=int(label_id),
            view=widget.currentViewName,
            axis=axis,
            slice_idx=int(slice_idx),
            delta_pixels=_count_delta(delta),
            model_key=widget.modelFamily._get_model_key(),
            params=widget.getUserParameters() or {},
            segment_id=seg_id,
            segmentation_id=seg.GetID(),
            volume_id=vol.GetID())
    return delta


def _spx_apply_direct(seg, seg_id, axis, slice_idx, combined_mask2d,
                       row_offset, col_offset, additive, ijk_to_ras_mat):
    """Write combined mask directly into the segment's binary labelmap numpy array.

    Uses the same coordinate mapping as _spx_read_native_slice: maps the labelmap's
    (0,0,0) voxel through its world matrix into reference-volume IJK to find where
    the labelmap sits, then offsets the modifier into the labelmap-local array.

    Returns True and fires Modified events on success.
    Returns False when the modifier extends beyond the current labelmap bounding box
    (caller must fall back to SetBinaryLabelmapToSegment for the expansion).

    O(modifier_size) — no VTK allocation, no reallocation.
    """
    import vtkSegmentationCorePython as vsc
    from vtk.util.numpy_support import vtk_to_numpy

    seg_obj = seg.GetSegmentation().GetSegment(seg_id)
    if seg_obj is None:
        return False
    rep_name = vsc.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    lm = seg_obj.GetRepresentation(rep_name)
    if lm is None:
        return False
    scalars = lm.GetPointData().GetScalars()
    if scalars is None or scalars.GetNumberOfTuples() == 0:
        return False

    # Map labelmap local (0,0,0) → ref-vol IJK (mirrors _spx_read_native_slice)
    lm_to_world = vtk.vtkMatrix4x4()
    lm.GetImageToWorldMatrix(lm_to_world)
    ref_ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras_mat, ref_ras_to_ijk)
    origin_ras = [0.0, 0.0, 0.0, 0.0]
    lm_to_world.MultiplyPoint([0.0, 0.0, 0.0, 1.0], origin_ras)
    ref_ijk_o  = [0.0, 0.0, 0.0, 0.0]
    ref_ras_to_ijk.MultiplyPoint(origin_ras, ref_ijk_o)
    ref_i = int(round(ref_ijk_o[0]))
    ref_j = int(round(ref_ijk_o[1]))
    ref_k = int(round(ref_ijk_o[2]))

    ext = lm.GetExtent()                  # (li0, li1, lj0, lj1, lk0, lk1)
    li0, li1, lj0, lj1, lk0, lk1 = ext
    ni_s = li1 - li0 + 1
    nj_s = lj1 - lj0 + 1
    nk_s = lk1 - lk0 + 1
    nr, nc = combined_mask2d.shape

    if axis == 0:   # Red/axial: K fixed, rows=J, cols=I
        arr_k = (slice_idx - ref_k) - lk0
        if not (0 <= arr_k < nk_s):
            return False
        lm_j0 = (row_offset - ref_j) - lj0
        lm_i0 = (col_offset - ref_i) - li0
        if lm_j0 < 0 or lm_j0 + nr > nj_s or lm_i0 < 0 or lm_i0 + nc > ni_s:
            return False
        arr = vtk_to_numpy(scalars).reshape(nk_s, nj_s, ni_s)
        target = arr[arr_k, lm_j0:lm_j0 + nr, lm_i0:lm_i0 + nc]
    elif axis == 1:  # Green/coronal: J fixed, rows=K, cols=I
        arr_j = (slice_idx - ref_j) - lj0
        if not (0 <= arr_j < nj_s):
            return False
        lm_k0 = (row_offset - ref_k) - lk0
        lm_i0 = (col_offset - ref_i) - li0
        if lm_k0 < 0 or lm_k0 + nr > nk_s or lm_i0 < 0 or lm_i0 + nc > ni_s:
            return False
        arr = vtk_to_numpy(scalars).reshape(nk_s, nj_s, ni_s)
        target = arr[lm_k0:lm_k0 + nr, arr_j, lm_i0:lm_i0 + nc]
    else:            # Yellow/sagittal: I fixed, rows=K, cols=J
        arr_i = (slice_idx - ref_i) - li0
        if not (0 <= arr_i < ni_s):
            return False
        lm_k0 = (row_offset - ref_k) - lk0
        lm_j0 = (col_offset - ref_j) - lj0
        if lm_k0 < 0 or lm_k0 + nr > nk_s or lm_j0 < 0 or lm_j0 + nc > nj_s:
            return False
        arr = vtk_to_numpy(scalars).reshape(nk_s, nj_s, ni_s)
        target = arr[lm_k0:lm_k0 + nr, lm_j0:lm_j0 + nc, arr_i]

    m = combined_mask2d.astype(np.uint8)
    if additive:
        target |= m
    else:
        target &= ~m

    scalars.Modified()
    seg.GetSegmentation().Modified()
    return True


def _spx_preallocate_full_labelmap(seg, seg_id, vol, ijk_to_ras_mat):
    """Pre-allocate the segment's binary labelmap to the full reference-volume extent.

    Uses seg_obj.AddRepresentation to set a fresh full-size vtkOrientedImageData
    directly on the segment — bypassing SetBinaryLabelmapToSegment and all its
    Modified event chains that would otherwise cause Slicer to reset or re-trim the
    labelmap before the first SPX stroke.

    Existing painted data is copied into the new full-size buffer.  No Modified()
    events are fired here; the first _spx_apply_direct call provides display
    notification (scalars.Modified + seg.GetSegmentation().Modified).

    After this call, _spx_apply_direct always succeeds for any view/slice, so
    SetBinaryLabelmapToSegment is never called during annotation strokes.
    """
    import vtkSegmentationCorePython as vsc
    from vtk.util.numpy_support import vtk_to_numpy

    seg_obj = seg.GetSegmentation().GetSegment(seg_id)
    if seg_obj is None:
        return
    rep_name = vsc.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()

    dims = vol.GetImageData().GetDimensions()   # (ni, nj, nk)
    ni, nj, nk = dims

    existing_lm = seg_obj.GetRepresentation(rep_name)
    if existing_lm is not None:
        ext = existing_lm.GetExtent()
        li0, li1, lj0, lj1, lk0, lk1 = ext
        if li1 - li0 + 1 >= ni and lj1 - lj0 + 1 >= nj and lk1 - lk0 + 1 >= nk:
            if SPX_DEBUG_TIMING:
                print(f'[SPX prealloc] already full: {dims}')
            return

    if SPX_DEBUG_TIMING:
        import time as _time
        _tp0 = _time.perf_counter()

    # Allocate a fresh full-size labelmap (all zeros, geometry = reference volume).
    new_lm = slicer.vtkOrientedImageData()
    new_lm.SetDimensions(ni, nj, nk)
    new_lm.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    new_scalars = new_lm.GetPointData().GetScalars()
    new_arr = vtk_to_numpy(new_scalars).reshape(nk, nj, ni)
    new_arr[:] = 0

    # Copy existing painted data into the correct position in the new buffer.
    if existing_lm is not None:
        old_scalars = existing_lm.GetPointData().GetScalars()
        if old_scalars and old_scalars.GetNumberOfTuples() > 0:
            lm_to_world = vtk.vtkMatrix4x4()
            existing_lm.GetImageToWorldMatrix(lm_to_world)
            ref_ras_to_ijk = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(ijk_to_ras_mat, ref_ras_to_ijk)
            origin_ras = [0.0, 0.0, 0.0, 0.0]
            lm_to_world.MultiplyPoint([0.0, 0.0, 0.0, 1.0], origin_ras)
            ref_ijk_o = [0.0, 0.0, 0.0, 0.0]
            ref_ras_to_ijk.MultiplyPoint(origin_ras, ref_ijk_o)
            ref_i = int(round(ref_ijk_o[0]))
            ref_j = int(round(ref_ijk_o[1]))
            ref_k = int(round(ref_ijk_o[2]))
            ext = existing_lm.GetExtent()
            li0, li1, lj0, lj1, lk0, lk1 = ext
            ni_s = li1 - li0 + 1
            nj_s = lj1 - lj0 + 1
            nk_s = lk1 - lk0 + 1
            old_arr = vtk_to_numpy(old_scalars).reshape(nk_s, nj_s, ni_s)
            dst_i0 = ref_i + li0; dst_j0 = ref_j + lj0; dst_k0 = ref_k + lk0
            si0 = max(0, dst_i0); si1 = min(ni, dst_i0 + ni_s)
            sj0 = max(0, dst_j0); sj1 = min(nj, dst_j0 + nj_s)
            sk0 = max(0, dst_k0); sk1 = min(nk, dst_k0 + nk_s)
            if si0 < si1 and sj0 < sj1 and sk0 < sk1:
                new_arr[sk0:sk1, sj0:sj1, si0:si1] = old_arr[
                    sk0 - dst_k0:sk1 - dst_k0,
                    sj0 - dst_j0:sj1 - dst_j0,
                    si0 - dst_i0:si1 - dst_i0]

    # Geometry: new labelmap spans the entire reference volume.
    # ijk_to_ras_mat maps (0,0,0) → RAS origin, same spacing and directions.
    new_lm.SetGeometryFromImageToWorldMatrix(ijk_to_ras_mat)
    new_scalars.Modified()

    # Set directly on the segment — bypasses SetBinaryLabelmapToSegment and its
    # Modified chain so Slicer cannot reset or re-trim the labelmap.
    # No seg.GetSegmentation().Modified() here; the first _spx_apply_direct
    # stroke fires the display notification.
    seg_obj.AddRepresentation(rep_name, new_lm)

    if SPX_DEBUG_TIMING:
        _tp1 = _time.perf_counter()
        print(f'[SPX prealloc] AddRepresentation total={(_tp1-_tp0)*1000:.0f}ms  '
              f'lm_dims={(ni, nj, nk)}')


def apply_spx_labels_batch(widget, label_ids, combined_mask2d,
                           axis, slice_idx, additive, view_name,
                           row_offset=0, col_offset=0,
                           ijk_to_ras_mat=None, modifier_box=None):
    """Apply a pre-built combined SPX mask in one modifySelectedSegmentByLabelmap call.

    combined_mask2d may be a bbox-cropped sub-window; row_offset / col_offset give its
    position within the full slice.  Recording still emits one event per label.

    ijk_to_ras_mat: optional pre-fetched vtkMatrix4x4 (skips GetIJKToRASMatrix per call).
    modifier_box:   optional [vtkOrientedImageData] for modifier reuse across strokes.
                    modifier_box[0] is used as the initial modifier and updated in-place.
    """
    editor = widget.logic.get_segment_editor()
    if editor is None:
        return
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    vol    = editor.sourceVolumeNode()
    if not (seg and seg_id and vol):
        return
    if SPX_DEBUG_TIMING:
        import time as _time
        _t0 = _time.perf_counter()

    # Fast path: write directly into the segment's binary labelmap numpy array.
    # O(modifier_size) — no VTK reallocation, no undo serialization.
    # Works whenever the modifier fits within the current labelmap bounding box.
    # Falls back to SetBinaryLabelmapToSegment only when new territory is covered
    # (which triggers a VTK reallocation proportional to the new union extent).
    direct_ok = _spx_apply_direct(
        seg, seg_id, axis, int(slice_idx), combined_mask2d,
        row_offset, col_offset, additive, ijk_to_ras_mat)

    if SPX_DEBUG_TIMING:
        _t1 = _time.perf_counter()

    if not direct_ok:
        # Modifier extends beyond existing labelmap bbox — fall back to VTK API.
        seg_logic = slicer.modules.segmentations.logic()
        modifier_in = modifier_box[0] if modifier_box is not None else None
        modifier = _build_slice_modifier_labelmap(
            vol, axis, int(slice_idx), combined_mask2d,
            row_offset=row_offset, col_offset=col_offset,
            ijk_to_ras_mat=ijk_to_ras_mat, modifier=modifier_in)
        if modifier_box is not None:
            modifier_box[0] = modifier
        merge_mode = seg_logic.MODE_MERGE_MAX if additive else seg_logic.MODE_MERGE_MIN
        ok = seg_logic.SetBinaryLabelmapToSegment(modifier, seg, seg_id, merge_mode)
        if not ok:
            import logging
            logging.warning(f'[SPX] SetBinaryLabelmapToSegment failed: seg_id={seg_id!r}')
            return

    delta_pixels = int(combined_mask2d.astype(bool).sum())

    if SPX_DEBUG_TIMING:
        _t2 = _time.perf_counter()
        import vtkSegmentationCorePython as _vsc
        _seg_lm = seg.GetSegmentation().GetSegment(seg_id).GetRepresentation(
            _vsc.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName())
        _lm_dims = _seg_lm.GetDimensions() if _seg_lm else None
        _lm_info = (f'lm_dims={_lm_dims} ({_lm_dims[0]*_lm_dims[1]*_lm_dims[2]/1e6:.1f}M vox)'
                    if _lm_dims else 'no_lm')
        _path = 'direct' if direct_ok else 'SetBinaryLabelmapToSegment'
        print(f'  [apply_batch] {_path}={(_t2-_t0)*1000:.1f}ms  {_lm_info}')

    if widget._recorder.is_active:
        model_key = widget.modelFamily._get_model_key()
        params    = widget.getUserParameters() or {}
        for i, label_id in enumerate(label_ids):
            widget._recorder.record_spx_fill(
                additive=additive,
                label_id=int(label_id),
                view=view_name,
                axis=axis,
                slice_idx=int(slice_idx),
                delta_pixels=delta_pixels if i == 0 else 0,
                model_key=model_key,
                params=params,
                segment_id=seg_id,
                segmentation_id=seg.GetID(),
                volume_id=vol.GetID())


def fill_hole_2d(widget):
    """Fill enclosed holes in the active segment on the current slice (2D only).

    Uses scipy.ndimage.binary_fill_holes then writes through the Segment Editor
    effect API so Slicer's undo stack captures the operation.
    """
    editor = widget.logic.get_segment_editor()
    if editor is None:
        return None
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    vol    = editor.sourceVolumeNode()
    if not (seg and seg_id and vol):
        slicer.util.warningDisplay('Fill Hole: no active segment or volume.')
        return None

    viewName = widget.currentViewName
    axis = VIEW_TO_AXIS.get(viewName, 2)
    lm = slicer.app.layoutManager()
    sw = lm.sliceWidget(viewName)
    sliceNode = sw.mrmlSliceNode()
    sliceToRAS = sliceNode.GetSliceToRAS()
    ras = [sliceToRAS.GetElement(r, 3) for r in range(3)]
    rasToIjk = vtk.vtkMatrix4x4()
    vol.GetRASToIJKMatrix(rasToIjk)
    ijk = rasToIjk.MultiplyPoint(ras + [1])
    slice_idx = int(round(ijk[AXIS_TO_IJK_COMPONENT[axis]]))

    current = _read_segment_slice(seg, seg_id, vol, axis, slice_idx)

    try:
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(current)
    except ImportError:
        slicer.util.warningDisplay('Fill Hole requires scipy. Install it to use this tool.')
        return None

    if not np.any(filled != current):
        return None

    editor.saveStateForUndo()
    modifier = _build_slice_modifier_labelmap(vol, axis, slice_idx, filled.astype(np.uint8))
    from slicer import qSlicerSegmentEditorAbstractEffect as _Eff
    effect = editor.effectByName('Paint')
    if effect is None:
        return None
    delta = effect.modifySelectedSegmentByLabelmap(modifier, _Eff.ModificationModeSet)

    if widget._recorder.is_active:
        widget._recorder.record_fill_hole(
            view=viewName, axis=axis,
            slice_idx=int(slice_idx), delta_pixels=_count_delta(delta),
            segment_id=seg_id, segmentation_id=seg.GetID(),
            volume_id=vol.GetID())
    return delta


def compute_spx_boundary(widget):
    """Compute SPX superpixel boundary pixels for the current slice.

    Returns ``(boundary_uint8_2d, axis, slice_idx)``.

    Raises ValueError with a user-readable message when the boundary cannot be
    computed (wrong family, no model, no volume, bad params).
    """
    modelFamily = widget.modelFamily
    if not isinstance(modelFamily, SPXModelFamily):
        raise ValueError("Please select an SPX model family first.")
    if not modelFamily.model:
        raise ValueError("Please confirm a model first.")

    volumeNode = widget.ui.sourceVolumeSelector.currentNode()
    if not volumeNode:
        raise ValueError("Please select an image volume first.")

    viewName = widget.currentViewName
    axis = VIEW_TO_AXIS.get(viewName, 2)
    lm = slicer.app.layoutManager()
    sw = lm.sliceWidget(viewName)
    sliceNode = sw.mrmlSliceNode()
    sliceToRAS = sliceNode.GetSliceToRAS()
    ras = [sliceToRAS.GetElement(r, 3) for r in range(3)]
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    ijk = rasToIjk.MultiplyPoint(ras + [1])
    slice_idx = int(round(ijk[AXIS_TO_IJK_COMPONENT[axis]]))

    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    if volumeArray is None:
        raise ValueError("Volume has no image data.")
    img = get_slice_from_volume(volumeArray, axis, slice_idx)
    params = widget.getUserParameters()
    if params is None:
        raise ValueError("Invalid model parameters.")
    labels = modelFamily.on_expand(
        volume_node=volumeNode, axis=axis, slice_idx=slice_idx,
        img=img, **params)
    if labels is None:
        raise ValueError("SPX model returned no labels for this slice.")

    return spx_boundary_mask(labels), axis, slice_idx


def compute_spx_boundary_for_volume(widget, volume, boundary_node):
    """Compute SPX boundaries for the current slice and write into boundary_node.

    boundary_node is a vtkMRMLLabelMapVolumeNode.  The full volume geometry is
    initialised on first call; subsequent calls update only the target slice so
    switching slices is cheap.
    """
    try:
        boundary_2d, axis, slice_idx = compute_spx_boundary(widget)
    except ValueError as exc:
        log.warning('compute_spx_boundary_for_volume: %s', exc)
        return

    vol_arr = slicer.util.arrayFromVolume(volume)

    # Allocate or re-allocate boundary node to match source volume geometry.
    existing = boundary_node.GetImageData()
    if existing is None or existing.GetDimensions() != volume.GetImageData().GetDimensions():
        zeros = np.zeros(vol_arr.shape, dtype=np.uint8)
        slicer.util.updateVolumeFromArray(boundary_node, zeros)
        mat = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(mat)
        boundary_node.SetIJKToRASMatrix(mat)
        if boundary_node.GetDisplayNode() is None:
            boundary_node.CreateDefaultDisplayNodes()

    arr = slicer.util.arrayFromVolume(boundary_node).copy()
    m = boundary_2d.astype(np.uint8)
    write_slice_to_volume(arr, m, axis, slice_idx)
    slicer.util.updateVolumeFromArray(boundary_node, arr)
