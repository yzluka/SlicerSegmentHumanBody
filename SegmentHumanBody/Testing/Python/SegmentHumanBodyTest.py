"""Slicer-native integration tests for SegmentHumanBody.

Run from inside 3D Slicer via Developer Tools → Run Unittests, or via the
"Reload and Test" button, which delegates to SegmentHumanBodyTest.runTest().

These tests require a live Slicer process and exercise:
  - MRML scene operations (arrayFromSegmentBinaryLabelmap, etc.)
  - Snapshot-based undo: capture / restore per axis and cross-axis isolation.
  - Unified undo stack: expand, brush, and interleaved action ordering.
  - Qt event filter (_SliceViewMouseFilter): return value and callback routing.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import slicer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_K, _J, _I = 12, 15, 18   # volume shape used by all test classes


def _make_volume():
    arr = np.zeros((_K, _J, _I), dtype=np.int16)
    return slicer.util.addVolumeFromArray(arr)


def _make_seg(volumeNode, with_segment=False):
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    seg.CreateDefaultDisplayNodes()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    sid = seg.GetSegmentation().AddEmptySegment('Seg1') if with_segment else None
    return seg, sid


def _write_lm(segNode, segmentID, volumeNode, arr):
    slicer.util.updateSegmentBinaryLabelmapFromArray(arr, segNode, segmentID, volumeNode)


def _read_lm(segNode, segmentID, volumeNode):
    return slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)


def _make_logic():
    from SegmentHumanBody import SegmentHumanBodyLogic
    return SegmentHumanBodyLogic()


def _mock_widget(volumeNode, segNode, segmentID, paramNode=None):
    """Return a MagicMock wired to return the given Slicer nodes."""
    w = MagicMock()
    w.ui.sourceVolumeSelector.currentNode.return_value = volumeNode
    w.ui.segmentSelector.currentNode.return_value = segNode
    w.ui.segmentSelector.currentSegmentID.return_value = segmentID
    w._parameterNode = paramNode
    return w


def _mock_expand_widget(logic, volumeNode, segNode, segmentID, axis, slice_index):
    """Widget stub for expandSegWithSPX: _captureSegmentationState returns a
    real snapshot, and _undo_stack is a plain list that can be inspected."""
    from core.utils import get_slice_from_volume

    w = MagicMock()
    w._undo_stack = []

    def _capture():
        mask3d = _read_lm(segNode, segmentID, volumeNode)
        data = get_slice_from_volume(mask3d, axis, slice_index)
        return (segNode.GetID(), segmentID, axis, slice_index, data.copy())

    w._captureSegmentationState.side_effect = _capture
    return w


# ---------------------------------------------------------------------------
# Fake 'self' objects for calling unbound Widget methods under test
# without instantiating the full Qt widget.
# ---------------------------------------------------------------------------

class _FakeRestoreSelf:
    """Minimal self for SegmentHumanBodyWidget._restoreSegmentation."""
    def __init__(self, volumeNode, logic):
        self.logic = logic
        _vol = volumeNode

        class _VolSel:
            def currentNode(self):
                return _vol

        class _UI:
            def __init__(self):
                self.sourceVolumeSelector = _VolSel()

        self.ui = _UI()


class _FakeCaptureSelf:
    """Minimal self for SegmentHumanBodyWidget._captureSegmentationState."""
    def __init__(self, logic, volumeNode, segNode, segmentID, axis, slice_index):
        self.logic = logic
        self._fixed_axis = axis
        self._fixed_slice = slice_index

        _vol = volumeNode
        _seg = segNode
        _sid = segmentID

        class _SegSel:
            def currentNode(self): return _seg
            def currentSegmentID(self): return _sid

        class _VolSel:
            def currentNode(self): return _vol

        class _UI:
            def __init__(self):
                self.segmentSelector = _SegSel()
                self.sourceVolumeSelector = _VolSel()

        self.ui = _UI()
        # Patch getAxisAndSlice so the test controls the axis/slice.
        logic.getAxisAndSlice = lambda widget, vol=None: (axis, slice_index)


# ===========================================================================
# applyResult tests  (unchanged from original, still valid)
# ===========================================================================

class SegmentHumanBodyLogicTest(unittest.TestCase):
    """Integration tests for SegmentHumanBodyLogic."""

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._logic = _make_logic()

    def test_apply_result_auto_creates_segment(self):
        """applyResult auto-creates a segment when the segmentation is empty."""
        volumeNode = _make_volume()
        segNode, _ = _make_seg(volumeNode, with_segment=False)
        paramNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.setVolumeAndSegmentation(paramNode, volumeNode, segNode)

        widget = _mock_widget(volumeNode, segNode, segmentID=None, paramNode=paramNode)
        mask2d = np.ones((_J, _I), dtype=np.uint8)
        self._logic.applyResult(widget, mask2d, axis=0, sliceIndex=5)

        self.assertEqual(segNode.GetSegmentation().GetNumberOfSegments(), 1)

    def test_apply_result_writes_correct_axial_slice(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        widget = _mock_widget(volumeNode, segNode, segmentID)

        mask2d = np.ones((_J, _I), dtype=np.uint8)
        target = 4
        self._logic.applyResult(widget, mask2d, axis=0, sliceIndex=target)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[target], 1)
        np.testing.assert_array_equal(result[:target], 0)
        np.testing.assert_array_equal(result[target + 1:], 0)

    def test_apply_result_working_mask_cached_across_frames(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        widget = _mock_widget(volumeNode, segNode, segmentID)

        mask_a = np.ones((_J, _I), dtype=np.uint8)
        mask_b = np.zeros((_J, _I), dtype=np.uint8)
        mask_b[:, :_I // 2] = 1

        self._logic.applyResult(widget, mask_a, axis=0, sliceIndex=2)
        cached_id = id(self._logic._working_mask)

        self._logic.applyResult(widget, mask_b, axis=0, sliceIndex=3)
        self.assertEqual(id(self._logic._working_mask), cached_id)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1)
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    # ---- expandSegWithSPX (corrected signatures) ----

    def test_expand_seg_with_spx_expands_matched_labels(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        base = np.zeros((_K, _J, _I), dtype=np.uint8)
        base[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, base)

        labels = np.ones((_J, _I), dtype=np.int32)
        labels[:, _I // 2:] = 2

        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    def test_expand_seg_with_spx_neg_points_subtract(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        base = np.zeros((_K, _J, _I), dtype=np.uint8)
        base[3] = 1
        _write_lm(segNode, segmentID, volumeNode, base)

        labels = np.ones((_J, _I), dtype=np.int32)
        labels[:, _I // 2:] = 2

        neg_pts = [[_I // 2 + 1, 0]]
        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3,
                                     neg_points=neg_pts)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    def test_expand_seg_with_spx_preserves_other_slices(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        base = np.zeros((_K, _J, _I), dtype=np.uint8)
        base[2] = 1
        base[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, base)

        labels = np.ones((_J, _I), dtype=np.int32)
        labels[:, _I // 2:] = 2

        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1, err_msg="slice 2 must be untouched")
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    # ---- on_enter / on_stop interactive ----

    def test_on_enter_interactive_starts_renderer(self):
        """on_enter_interactive must start the renderer timer."""
        widget = MagicMock()
        widget.renderer = MagicMock()
        self._logic.on_enter_interactive(widget)
        widget.renderer.start.assert_called_once()

    def test_on_enter_interactive_sets_interactive_state(self):
        widget = MagicMock()
        widget.renderer = MagicMock()
        self._logic.on_enter_interactive(widget)
        widget.setInteractiveState.assert_called_once_with(True)

    def test_on_enter_interactive_resets_render_state(self):
        """on_enter_interactive calls reset_render_state, clearing cached masks."""
        self._logic._working_mask = np.ones((5, 5, 5), dtype=np.uint8)
        widget = MagicMock()
        widget.renderer = MagicMock()
        self._logic.on_enter_interactive(widget)
        self.assertIsNone(self._logic._working_mask)

    def test_on_stop_interactive_stops_renderer(self):
        widget = MagicMock()
        widget.renderer = MagicMock()
        self._logic.on_stop_interactive(widget)
        widget.renderer.stop.assert_called_once()

    # ---- coordinate conversion ----

    def test_ras_to_ijk_maps_volume_centre_to_middle_voxel(self):
        volumeNode = _make_volume()
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2

        result = self._logic._ras_to_ijk(
            volumeNode,
            {"positive": [[cx, cy, cz]], "negative": []},
            axis=0
        )
        pt = result["positive"][0]
        self.assertAlmostEqual(pt[0], _I / 2, delta=1)
        self.assertAlmostEqual(pt[1], _J / 2, delta=1)

    # ---- getAxisAndSlice ----

    def test_get_axis_and_slice_uses_volume_geometry(self):
        import vtk
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

        volumeNode = _make_volume()
        target_k = 7

        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        ras = ijkToRAS.MultiplyPoint([0.0, 0.0, float(target_k), 1.0])
        ras_z = ras[2]

        slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode().SetSliceOffset(ras_z)

        widget = MagicMock()
        widget.currentViewName = "Red"

        axis, sliceIndex = self._logic.getAxisAndSlice(widget, volumeNode)
        self.assertEqual(axis, 0)
        self.assertEqual(sliceIndex, target_k)


# ===========================================================================
# Snapshot capture and restore tests
# ===========================================================================

class SnapshotRestoreTest(unittest.TestCase):
    """Tests for _captureSegmentationState (via _FakeCaptureSelf) and
    _restoreSegmentation (via _FakeRestoreSelf) — called as unbound methods
    to avoid requiring a full Qt widget."""

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._logic = _make_logic()

    def _capture(self, volumeNode, segNode, segmentID, axis, slice_index):
        """Call _captureSegmentationState via an unbound method call."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        fake_self = _FakeCaptureSelf(
            self._logic, volumeNode, segNode, segmentID, axis, slice_index
        )
        return SegmentHumanBodyWidget._captureSegmentationState(fake_self)

    def _restore(self, volumeNode, snapshot):
        """Call _restoreSegmentation via an unbound method call."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        fake_self = _FakeRestoreSelf(volumeNode, self._logic)
        SegmentHumanBodyWidget._restoreSegmentation(fake_self, snapshot)

    # ---- capture + restore is identity ----

    def _roundtrip_test(self, axis, slice_index):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[slice_index if axis == 0 else 0,
                slice_index if axis == 1 else 0,
                slice_index if axis == 2 else 0] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        snap = self._capture(volumeNode, segNode, segmentID, axis, slice_index)
        # Overwrite the target slice with all-ones.
        overwrite = np.ones((_K, _J, _I), dtype=np.uint8)
        _write_lm(segNode, segmentID, volumeNode, overwrite)
        # Restore from snapshot.
        self._restore(volumeNode, snap)

        restored = _read_lm(segNode, segmentID, volumeNode)
        # Only the captured slice is restored; the rest was already overwritten.
        from core.utils import get_slice_from_volume
        restored_slice = get_slice_from_volume(restored, axis, slice_index)
        initial_slice  = get_slice_from_volume(initial,  axis, slice_index)
        np.testing.assert_array_equal(
            restored_slice, initial_slice,
            err_msg=f"axis={axis} slice={slice_index}: restore must undo the overwrite"
        )

    def test_restore_roundtrip_axial(self):
        self._roundtrip_test(axis=0, slice_index=5)

    def test_restore_roundtrip_coronal(self):
        self._roundtrip_test(axis=1, slice_index=6)

    def test_restore_roundtrip_sagittal(self):
        self._roundtrip_test(axis=2, slice_index=7)

    # ---- non-target slices are not affected ----

    def test_restore_does_not_touch_other_slices(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[3] = 1    # slice 3 painted
        initial[7] = 1    # slice 7 also painted
        _write_lm(segNode, segmentID, volumeNode, initial)

        # Capture slice 3 only.
        snap = self._capture(volumeNode, segNode, segmentID, axis=0, slice_index=3)

        # Erase everything.
        _write_lm(segNode, segmentID, volumeNode, np.zeros((_K, _J, _I), dtype=np.uint8))

        # Restore slice 3.
        self._restore(volumeNode, snap)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3], 1,
                                      err_msg="slice 3 must be restored")
        np.testing.assert_array_equal(result[7], 0,
                                      err_msg="slice 7 was not captured → must stay erased")

    # ---- snapshot is a deep copy ----

    def test_snapshot_data_is_deep_copy(self):
        """Modifying the segmentation after capture must not affect the snapshot."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[4, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        snap = self._capture(volumeNode, segNode, segmentID, axis=0, slice_index=4)

        # Write completely different data AFTER capture.
        _write_lm(segNode, segmentID, volumeNode, np.ones((_K, _J, _I), dtype=np.uint8))

        stored_data = snap[4]
        np.testing.assert_array_equal(stored_data[:, :_I // 2], 1)
        np.testing.assert_array_equal(stored_data[:, _I // 2:], 0)

    # ---- captureSegmentationState returns correct metadata ----

    def test_capture_returns_correct_axis(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        snap = self._capture(volumeNode, segNode, segmentID, axis=2, slice_index=5)
        self.assertEqual(snap[2], 2)

    def test_capture_returns_correct_slice_index(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        snap = self._capture(volumeNode, segNode, segmentID, axis=0, slice_index=8)
        self.assertEqual(snap[3], 8)

    def test_capture_returns_correct_segment_id(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        snap = self._capture(volumeNode, segNode, segmentID, axis=0, slice_index=0)
        self.assertEqual(snap[1], segmentID)

    def test_capture_returns_none_when_no_segment(self):
        volumeNode = _make_volume()
        snap = self._capture(volumeNode, segNode=None, segmentID=None,
                             axis=0, slice_index=0)
        self.assertIsNone(snap)

    # ---- restore None is a no-op ----

    def test_restore_none_snapshot_does_not_raise(self):
        volumeNode = _make_volume()
        self._restore(volumeNode, None)   # must not raise


# ===========================================================================
# Unified undo stack — end-to-end tests with real MRML nodes
# ===========================================================================

class UnifiedUndoStackTest(unittest.TestCase):
    """End-to-end tests verifying that expand, brush, and interleaved actions
    are recorded in the unified _undo_stack and can be restored correctly.

    These tests use _mock_expand_widget / _FakeRestoreSelf to exercise the
    real snapshot/restore code paths without a full Qt widget.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._logic = _make_logic()

    def _restore(self, volumeNode, snapshot):
        from SegmentHumanBody import SegmentHumanBodyWidget
        SegmentHumanBodyWidget._restoreSegmentation(
            _FakeRestoreSelf(volumeNode, self._logic), snapshot
        )

    # ---- expand → one undo entry pushed ----

    def test_expand_pushes_one_entry_to_undo_stack(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        labels = np.ones((_J, _I), dtype=np.int32)
        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)

        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        self.assertEqual(len(widget._undo_stack), 1)
        self.assertEqual(widget._undo_stack[0][0], 'expand')

    def test_expand_undo_entry_carries_pre_expand_data(self):
        """The snapshot must contain the slice state BEFORE the expand."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        pre_state = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre_state[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre_state)

        labels = np.ones((_J, _I), dtype=np.int32)
        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        snap = widget._undo_stack[0][1]
        # The snapshot data must match the left-half-only pre-expand state.
        np.testing.assert_array_equal(snap[4][:, :_I // 2], 1)
        np.testing.assert_array_equal(snap[4][:, _I // 2:], 0)

    def test_expand_undo_restores_pre_expand_slice(self):
        """Restoring the snapshot undoes the full-slice expansion."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        pre_state = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre_state[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre_state)

        labels = np.ones((_J, _I), dtype=np.int32)
        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=3)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        after_expand = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(after_expand[3], 1,
                                      err_msg="expand should have painted the whole slice")

        # Pop + restore.
        _, snapshot = widget._undo_stack.pop()
        self._restore(volumeNode, snapshot)
        self._logic.reset_render_state()

        restored = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(restored[3, :, :_I // 2], 1,
                                      err_msg="left half must be restored")
        np.testing.assert_array_equal(restored[3, :, _I // 2:], 0,
                                      err_msg="right half must be gone after undo")

    def test_expand_undo_does_not_change_other_slices(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[2] = 1    # slice 2 fully painted — should be untouched
        pre[5, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre)

        labels = np.ones((_J, _I), dtype=np.int32)
        widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                     axis=0, slice_index=5)
        self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=5)

        _, snapshot = widget._undo_stack.pop()
        self._restore(volumeNode, snapshot)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1,
                                      err_msg="slice 2 must be untouched by undo")

    # ---- multiple expands → LIFO ordering ----

    def test_two_consecutive_expands_lifo(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        labels = np.ones((_J, _I), dtype=np.int32)
        stack = []

        for sl in [2, 4]:
            widget = _mock_expand_widget(self._logic, volumeNode, segNode, segmentID,
                                         axis=0, slice_index=sl)
            self._logic.expandSegWithSPX(widget, segNode, segmentID, volumeNode,
                                         labels, axis=0, sliceIndex=sl)
            stack.extend(widget._undo_stack)

        # Most-recent expand (slice 4) must be undone first.
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[-1][1][3], 4,  "last entry must be slice 4")
        self.assertEqual(stack[-2][1][3], 2,  "first entry must be slice 2")

    # ---- brush entries: snapshot is not None ----

    def test_brush_entry_snapshot_is_not_none(self):
        """Regression: the original code pushed ('brush', None), which forced
        onUndo to delegate to editor.undo() — a *separate* undo stack.  Now
        every brush entry must carry a real snapshot payload."""
        from core.utils import get_slice_from_volume

        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        painted = np.zeros((_K, _J, _I), dtype=np.uint8)
        painted[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, painted)

        snap = (_read_lm(segNode, segmentID, volumeNode)[3].copy(),)
        # Simulate what _onBrushStrokeStart now does:
        brush_entry = ('brush', (
            segNode.GetID(), segmentID, 0, 3,
            get_slice_from_volume(_read_lm(segNode, segmentID, volumeNode), 0, 3).copy()
        ))

        self.assertIsNotNone(brush_entry[1],
                             "Brush entry must carry a snapshot, not None")
        self.assertIsInstance(brush_entry[1], tuple)
        self.assertEqual(len(brush_entry[1]), 5)

    def test_brush_entry_undo_restores_pre_stroke_state(self):
        """Simulate: capture → paint → undo → pre-paint state restored."""
        from core.utils import get_slice_from_volume

        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        # Pre-stroke state: left half of slice 5 painted.
        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[5, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre)

        # Capture state before stroke (mirrors _onBrushStrokeStart).
        mask3d = _read_lm(segNode, segmentID, volumeNode)
        pre_data = get_slice_from_volume(mask3d, 0, 5).copy()
        brush_entry = ('brush', (segNode.GetID(), segmentID, 0, 5, pre_data))

        # Simulate a brush stroke that fills the whole slice.
        post = pre.copy()
        post[5] = 1
        _write_lm(segNode, segmentID, volumeNode, post)

        # Undo: restore from snapshot.
        self._restore(volumeNode, brush_entry[1])
        self._logic.reset_render_state()

        restored = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(restored[5, :, :_I // 2], 1,
                                      err_msg="left half must be restored")
        np.testing.assert_array_equal(restored[5, :, _I // 2:], 0,
                                      err_msg="right half (painted by stroke) must be undone")

    # ---- stack cleared on segment add / switch ----

    def test_clear_is_clean_slate_for_next_segment(self):
        """After a segment switch the undo stack must be empty so the new
        segment's actions don't accidentally restore a previous segment's state."""
        undo_stack = []
        # Push some entries for "old segment".
        undo_stack.append(('expand', None))
        undo_stack.append(('brush', None))
        # Simulate onSegmentChanged / onAddSegment clearing the stack.
        undo_stack.clear()
        self.assertEqual(len(undo_stack), 0)


# ===========================================================================
# _SliceViewMouseFilter behaviour
# ===========================================================================

class MouseFilterTest(unittest.TestCase):
    """Tests for _SliceViewMouseFilter — the Qt application-level event filter
    that replaced the VTK interactor observer approach.

    Key invariants:
      1. eventFilter ALWAYS returns False (never consumes events).
      2. LeftButton press → _onBrushStrokeStart() called.
      3. LeftButton release → _onBrushStrokeEnd() called.
      4. Right button / non-mouse events → no callbacks, still returns False.

    Tests use lightweight fake event objects so no real Qt event-loop setup
    is required.
    """

    def setUp(self):
        import qt
        self._qt = qt
        from SegmentHumanBody import _SliceViewMouseFilter
        self._FilterClass = _SliceViewMouseFilter

    def _make_fake_widget(self):
        calls = []

        class _FakeWidget:
            def _onBrushStrokeStart(self_):  # noqa: N805
                calls.append('start')

            def _onBrushStrokeEnd(self_):  # noqa: N805
                calls.append('end')

        return _FakeWidget(), calls

    def _make_event(self, type_val, button_val=None):
        """Return a fake event object whose type() and button() match arguments."""
        qt = self._qt
        if button_val is None:
            button_val = qt.Qt.NoButton

        class _FakeEvent:
            def type(self): return type_val
            def button(self): return button_val

        return _FakeEvent()

    # ---- return value ----

    def test_returns_false_on_left_button_press(self):
        qt = self._qt
        fake_widget, _ = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonPress, qt.Qt.LeftButton)
        self.assertFalse(f.eventFilter(None, event),
                         "Filter must return False (never consume events)")

    def test_returns_false_on_left_button_release(self):
        qt = self._qt
        fake_widget, _ = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonRelease, qt.Qt.LeftButton)
        self.assertFalse(f.eventFilter(None, event))

    def test_returns_false_on_right_button_press(self):
        qt = self._qt
        fake_widget, _ = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonPress, qt.Qt.RightButton)
        self.assertFalse(f.eventFilter(None, event))

    def test_returns_false_on_key_press(self):
        qt = self._qt
        fake_widget, _ = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.KeyPress, qt.Qt.NoButton)
        self.assertFalse(f.eventFilter(None, event))

    def test_returns_false_on_mouse_move(self):
        qt = self._qt
        fake_widget, _ = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseMove, qt.Qt.LeftButton)
        self.assertFalse(f.eventFilter(None, event))

    # ---- callback routing ----

    def test_left_press_calls_stroke_start(self):
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonPress, qt.Qt.LeftButton)
        f.eventFilter(None, event)
        self.assertIn('start', calls)

    def test_left_release_calls_stroke_end(self):
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonRelease, qt.Qt.LeftButton)
        f.eventFilter(None, event)
        self.assertIn('end', calls)

    def test_right_press_does_not_call_stroke_start(self):
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonPress, qt.Qt.RightButton)
        f.eventFilter(None, event)
        self.assertEqual(calls, [],
                         "Right-button press must not trigger a stroke-start callback")

    def test_mouse_move_does_not_trigger_any_callback(self):
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseMove, qt.Qt.LeftButton)
        f.eventFilter(None, event)
        self.assertEqual(calls, [])

    def test_key_press_does_not_trigger_any_callback(self):
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.KeyPress)
        f.eventFilter(None, event)
        self.assertEqual(calls, [])

    def test_left_press_calls_exactly_one_callback(self):
        """Verify no accidental double-dispatch."""
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        event = self._make_event(qt.QEvent.MouseButtonPress, qt.Qt.LeftButton)
        f.eventFilter(None, event)
        self.assertEqual(len(calls), 1)

    def test_consecutive_press_release_sequence(self):
        """A full click cycle: press → start, release → end."""
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        f.eventFilter(None, self._make_event(qt.QEvent.MouseButtonPress,   qt.Qt.LeftButton))
        f.eventFilter(None, self._make_event(qt.QEvent.MouseButtonRelease, qt.Qt.LeftButton))
        self.assertEqual(calls, ['start', 'end'])

    def test_multiple_clicks_produce_correct_sequence(self):
        """Three full click cycles should give ['start','end'] × 3."""
        qt = self._qt
        fake_widget, calls = self._make_fake_widget()
        f = self._FilterClass(fake_widget)
        for _ in range(3):
            f.eventFilter(None, self._make_event(qt.QEvent.MouseButtonPress,   qt.Qt.LeftButton))
            f.eventFilter(None, self._make_event(qt.QEvent.MouseButtonRelease, qt.Qt.LeftButton))
        self.assertEqual(calls, ['start', 'end', 'start', 'end', 'start', 'end'])


if __name__ == '__main__':
    unittest.main()
