"""Business logic for the SegmentHumanBody Slicer module.

Separated from SegmentHumanBody.py so the Widget file stays focused on UI.
All Slicer / VTK / Qt imports live here; core/ remains pure-Python.
"""

import logging
import vtk
import slicer
import numpy as np
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
    # SPX boundary
    # -------------------------

    def compute_spx_boundary(self, widget):
        """Compute SPX superpixel boundary pixels for the current slice.

        Reuses the SPX label-map cache when available (no extra forward pass
        if the user is already in interactive mode or has expanded).  Falls
        back to running the model if the cache is empty.

        Returns ``(boundary_uint8_2d, axis, sliceIndex)``.

        Raises
        ------
        ValueError
            With a user-readable message when the boundary cannot be computed
            (wrong model family, no model confirmed, no volume, bad params).
        """
        modelFamily = widget.modelFamily
        if not isinstance(modelFamily, SPXModelFamily):
            raise ValueError("Please select an SPX model family first.")
        if not modelFamily.model:
            raise ValueError("Please confirm a model first (click 'Confirm Model').")

        volumeNode, _, _ = self._get_context(widget)
        if not volumeNode:
            raise ValueError("Please select an image volume first.")

        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)

        # Always go through on_expand so its cache key (which includes
        # img.shape) is validated against the current axis/slice.  Bypassing
        # it with a raw _cache_labels check causes a shape mismatch when the
        # user switches slice planes (e.g. Red → Green) after the model ran.
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        if volumeArray is None:
            raise ValueError("Volume has no image data.")
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)
        params = widget.getUserParameters()
        if params is None:
            raise ValueError("Invalid model parameters.")
        labels = modelFamily.on_expand(img=img, **params)

        if labels is None:
            raise ValueError("SPX model returned no labels for this slice.")

        return spx_boundary_mask(labels), axis, sliceIndex

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
        labels = call_if_exists(modelFamily, 'on_expand', img=img, **params)

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
