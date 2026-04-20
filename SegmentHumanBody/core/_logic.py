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
        # Render-skip: skip the write path when nothing has changed since the
        # last frame (same prompts, axis, slice, params).
        self._last_render_key = None

        # Numpy cache for the active segment; routes all reads/writes through
        # a single object so Slicer stays in sync.  Replaced lazily by
        # _get_tracker() when the segment identity changes.
        self._tracker: SegmentTracker | None = None

        # Frozen 3-D snapshot of the segment taken on the first confirmed prompt.
        # Lets every render recompute result = base + pos_labels − neg_labels
        # from scratch, so removing any point always reverts correctly.
        self._session_base: np.ndarray | None = None

        # Confirmed W/L values (set via "Apply Window/Level").  When set, each
        # slice is normalized to [0, 255] before reaching the model.  Volume
        # data is never modified.
        self._wl_window = None
        self._wl_level  = None

        # Most recent MaskChange from write_slice; picked up by the async
        # point-confirm flow (render fires → _capturePointChange reads this).
        self._last_change = None

    def setDefaultParameters(self, parameterNode):
        pass

    # -------------------------
    # Cache management
    # -------------------------

    def reset_render_state(self):
        """Clear the session snapshot and render key so the next render starts fresh.

        The tracker and W/L values are intentionally preserved — the tracker
        holds the active mask cache and W/L is a user preference that should
        survive segment changes.
        """
        self._last_render_key = None
        self._session_base = None

    def _get_tracker(self, segNode, segmentID, volumeNode) -> SegmentTracker:
        """Return the active ``SegmentTracker``, creating one when necessary.

        A new tracker is created whenever ``reset_render_state()`` has been
        called (sets ``_tracker = None``) or when the segment identity changes.
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
        """Force-reload the tracker and return the current slice as a before-state.

        Called at brush-stroke start so the widget can hold the pre-paint
        snapshot.  Drops the tracker cache first so the snapshot always
        reflects the latest MRML-committed state.

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
        tracker.sync()                                  # force fresh reload
        before_slice = tracker.get_slice(axis, idx).copy()
        return axis, idx, before_slice

    def commit_stroke(self, widget, axis, idx, before_slice, source='brush') -> 'MaskChange | None':
        """Record a brush stroke as a tracked delta via the single write path.

        Reads the after-state directly from Slicer (bypassing the cached
        ``_mask``), temporarily restores ``before_slice`` into ``_mask`` so
        that ``write_slice`` computes ``delta = after − before``, then calls
        ``write_slice`` which updates ``_mask`` to the after-state and pushes
        to Slicer.

        Returns the ``MaskChange`` to be stored in the widget's ``_history``,
        or ``None`` when the stroke produced no net change.
        """
        vol, seg, seg_id = self._get_context(widget)
        if not vol or not seg or not seg_id:
            return None
        tracker = self._get_tracker(seg, seg_id, vol)

        # Read the committed after-state from Slicer without touching _mask.
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(seg, seg_id, vol)
        if raw is None:
            log.warning('[Logic] commit_stroke: labelmap read returned None — stroke lost')
            return None
        after_slice = get_slice_from_volume(raw, axis, idx).copy()

        # Restore before-state in _mask so write_slice sees the right baseline.
        if tracker._mask is None:
            tracker._load()
        write_slice_to_volume(tracker._mask, before_slice, axis, idx)

        # Single write path: delta = after − before, updates _mask, pushes.
        return tracker.write_slice(axis, idx, after_slice, source=source)

    def reverse_change(self, widget, change) -> None:
        """Apply the inverse of *change* to the tracker and push to Slicer."""
        if change is None:
            return
        vol, seg, seg_id = self._get_context(widget)
        if not vol or not seg or not seg_id:
            return
        self._get_tracker(seg, seg_id, vol).reverse_delta(change)

    def end_session(self):
        """End the current annotation session.

        Clears the session and render key so the next render starts a fresh
        session with the current committed state as the new base.

        Slicer is NOT written to here because in the new design every render
        already commits directly — there is never a stale preview to flush.
        """
        self._session_base = None
        self._last_render_key = None

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

    # -------------------------
    # Render loop
    # -------------------------

    def onRender(self, modelFamily, widget):
        """Compute and commit the prompt result for the current slice.

        Called by ``_triggerRender()`` from point-confirmed / point-removed /
        slice-scroll event handlers and by ``ctrl.request_render()``.

        Design principle: **always commit**.  Every call writes the result
        through the ``SegmentTracker`` (cache + Slicer together).  There is no
        separate "preview" state — what Slicer shows is always the committed state.
        The ``_session_base`` snapshot (taken on the first prompt)
        lets us recompute any slice from scratch so that removing a point
        correctly reverts to base without any per-point snapshot.
        """
        if not modelFamily or not modelFamily.model:
            return

        parameterNode = widget._parameterNode
        posNode, negNode = self.getPromptNodes(parameterNode)

        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)

        # Only include control points that have been placed (PositionDefined).
        # The live placement cursor is in PositionPreview state and must not
        # drive a render — it is not yet a confirmed annotation.
        pos_points = [pos for (status, _, pos) in _node_records(posNode)
                      if status == POSITION_DEFINED]
        neg_points = [neg for (status, _, neg) in _node_records(negNode)
                      if status == POSITION_DEFINED]

        params = widget.getUserParameters()
        if params is None:
            log.warning("[onRender] Parameter parsing failed — skipping render")
            return

        render_key = (
            tuple(tuple(p) for p in pos_points),
            tuple(tuple(p) for p in neg_points),
            axis,
            sliceIndex,
            tuple(sorted(params.items())) if params else (),
        )

        if render_key == self._last_render_key:
            return

        # -----------------------------------------------------------------
        # No active prompts
        # -----------------------------------------------------------------
        if not pos_points and not neg_points:
            if self._session_base is not None:
                # Restore this slice to the session base (removes any
                # prompt-driven region that was committed while prompts existed).
                _, segNode, segmentID = self._get_context(widget)
                if segNode and segmentID:
                    self._get_tracker(segNode, segmentID, volumeNode).write_slice(
                        axis, sliceIndex,
                        get_slice_from_volume(self._session_base, axis, sliceIndex),
                        source='prompt',
                    )
                # End session once all markup nodes are empty.
                total = sum(n.GetNumberOfControlPoints() for n in (posNode, negNode) if n)
                if total == 0:
                    self._session_base = None
            self._last_render_key = render_key
            return

        # -----------------------------------------------------------------
        # Ensure a segment holder exists (create lazily on first prompt)
        # -----------------------------------------------------------------
        seg = widget.ui.segmentSelector.currentNode()
        seg_id = widget.ui.segmentSelector.currentSegmentID()
        if not seg or not seg_id:
            self._ensure_seg_and_segment(widget, volumeNode)

        _, segNode, segmentID = self._get_context(widget)
        if not segNode or not segmentID:
            return

        # -----------------------------------------------------------------
        # Start session on first prompt (snapshot committed state as base)
        # -----------------------------------------------------------------
        if self._session_base is None:
            self._session_base = self._get_tracker(segNode, segmentID, volumeNode).snapshot()

        # -----------------------------------------------------------------
        # Compute
        # -----------------------------------------------------------------
        scribbles_ijk = self.ras_to_ijk(volumeNode, {
            "positive": pos_points,
            "negative": neg_points,
        }, axis, slice_index=sliceIndex)

        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        if volumeArray is None:
            log.warning("[onRender] volume has no image data — skipping render")
            return
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)

        base_slice = get_slice_from_volume(self._session_base, axis, sliceIndex)

        result = call_if_exists(
            modelFamily, "onRender",
            img=img,
            pos_points=scribbles_ijk["positive"],
            neg_points=scribbles_ijk["negative"],
            base_mask=base_slice.copy(),
            **params
        )

        # -----------------------------------------------------------------
        # Commit — write result (or base) directly into tracker cache + Slicer
        # -----------------------------------------------------------------
        if result is None:
            log.warning('[Logic] onRender: model returned None — writing base slice')
        self._last_render_key = render_key
        self._last_change = self._get_tracker(segNode, segmentID, volumeNode).write_slice(
            axis, sliceIndex,
            result if result is not None else base_slice,
            source='prompt',
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

        return segNode, segmentID

    def applyResult(self, widget, mask2d, axis, sliceIndex):
        volumeNode, segNode, segmentID = self._get_context(widget)
        if not volumeNode or not segNode or not segmentID:
            return

        self._last_change = self._get_tracker(segNode, segmentID, volumeNode).write_slice(
            axis, sliceIndex, mask2d, source='prompt'
        )

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
        # End the current session so the next prompt render starts from the
        # post-expand committed state.
        self.end_session()

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
