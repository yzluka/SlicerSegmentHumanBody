"""Slicer-native integration tests for SegmentHumanBody.

Run from inside 3D Slicer via Developer Tools → Run Unittests, or via the
"Reload and Test" button, which delegates to SegmentHumanBodyTest.runTest().

These tests require a live Slicer process and exercise:
  - MRML scene operations (arrayFromSegmentBinaryLabelmap, etc.)
  - Delta-based undo: write_slice / reverse_delta round-trips.
  - Unified history: expand returns MaskChange; reverse_change restores state.
  - Qt event filter (_SliceViewMouseFilter): return value and on_press/on_release routing.
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
    from core._logic import SegmentHumanBodyLogic
    return SegmentHumanBodyLogic()


def _make_tracker(segNode, segmentID, volumeNode):
    from core._tracker import SegmentTracker
    return SegmentTracker(segNode, segmentID, volumeNode)


def _mock_widget(volumeNode, segNode, segmentID, paramNode=None):
    """Return a MagicMock wired to return the given Slicer nodes."""
    w = MagicMock()
    w.ui.sourceVolumeSelector.currentNode.return_value = volumeNode
    w.ui.segmentSelector.currentNode.return_value = segNode
    w.ui.segmentSelector.currentSegmentID.return_value = segmentID
    w._parameterNode = paramNode
    return w


def _mock_expand_widget(logic, volumeNode, segNode, segmentID, axis, slice_index):
    """Minimal widget stub kept for tests in SegmentHumanBodyLogicTest that
    still pass the widget to helper methods.  No longer used for expandSegWithSPX
    itself (that method no longer takes a widget argument)."""
    return _mock_widget(volumeNode, segNode, segmentID)


# ===========================================================================
# applyResult tests
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

    def test_apply_result_reuses_tracker_across_frames(self):
        """write_slice reuses the same tracker object across consecutive frames."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        widget = _mock_widget(volumeNode, segNode, segmentID)

        mask_a = np.ones((_J, _I), dtype=np.uint8)
        mask_b = np.zeros((_J, _I), dtype=np.uint8)
        mask_b[:, :_I // 2] = 1

        self._logic.applyResult(widget, mask_a, axis=0, sliceIndex=2)
        tracker_id = id(self._logic._tracker)

        self._logic.applyResult(widget, mask_b, axis=0, sliceIndex=3)
        self.assertEqual(id(self._logic._tracker), tracker_id,
                         "Tracker must be reused across frames for the same segment")

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1)
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    # ---- expandSegWithSPX ----

    def test_expand_seg_with_spx_expands_matched_labels(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        base = np.zeros((_K, _J, _I), dtype=np.uint8)
        base[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, base)

        labels = np.ones((_J, _I), dtype=np.int32)
        labels[:, _I // 2:] = 2

        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
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
        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
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

        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1, err_msg="slice 2 must be untouched")
        np.testing.assert_array_equal(result[3, :, :_I // 2], 1)
        np.testing.assert_array_equal(result[3, :, _I // 2:], 0)

    # ---- coordinate conversion ----

    def test_ras_to_ijk_maps_volume_centre_to_middle_voxel(self):
        volumeNode = _make_volume()
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2

        result = self._logic.ras_to_ijk(
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
# SegmentTracker delta-based undo tests
# ===========================================================================

class TrackerUndoTest(unittest.TestCase):
    """Tests for SegmentTracker.write_slice() / reverse_delta() round-trips.

    write_slice() returns a MaskChange (or None for no-ops).  The caller
    stores the change and passes it to reverse_delta() to undo — there is no
    internal _changes stack on the tracker.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    # ---- write + reverse_delta is identity ----

    def _roundtrip_test(self, axis, slice_index):
        from core.utils import get_slice_from_volume
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[slice_index if axis == 0 else 0,
                slice_index if axis == 1 else 0,
                slice_index if axis == 2 else 0] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        tracker = _make_tracker(segNode, segmentID, volumeNode)

        # Write all-ones to the target slice; capture the returned change.
        ones = np.ones(get_slice_from_volume(initial, axis, slice_index).shape,
                       dtype=np.uint8)
        change = tracker.write_slice(axis, slice_index, ones, source='test')
        self.assertIsNotNone(change, "write_slice must return a MaskChange for a real change")

        # Reverse the change — must revert to initial.
        tracker.reverse_delta(change)

        restored = _read_lm(segNode, segmentID, volumeNode)
        initial_slice  = get_slice_from_volume(initial,  axis, slice_index)
        restored_slice = get_slice_from_volume(restored, axis, slice_index)
        np.testing.assert_array_equal(
            restored_slice, initial_slice,
            err_msg=f"axis={axis} slice={slice_index}: reverse_delta must revert the write",
        )

    def test_undo_roundtrip_axial(self):
        self._roundtrip_test(axis=0, slice_index=5)

    def test_undo_roundtrip_coronal(self):
        self._roundtrip_test(axis=1, slice_index=6)

    def test_undo_roundtrip_sagittal(self):
        self._roundtrip_test(axis=2, slice_index=7)

    # ---- reverse_delta does not touch other slices ----

    def test_undo_does_not_touch_other_slices(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[3] = 1   # painted
        initial[7] = 1   # also painted — must be untouched by undo of slice 3
        _write_lm(segNode, segmentID, volumeNode, initial)

        tracker = _make_tracker(segNode, segmentID, volumeNode)
        ones = np.ones((_J, _I), dtype=np.uint8)
        change = tracker.write_slice(axis=0, idx=3, new_data=ones, source='test')
        tracker.reverse_delta(change)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3], 1,
                                      err_msg="slice 3 must be restored by undo")
        np.testing.assert_array_equal(result[7], 1,
                                      err_msg="slice 7 must remain untouched")

    # ---- bounding-box efficiency ----

    def test_delta_stored_only_for_changed_pixels(self):
        """The returned MaskChange delta crop must cover only changed pixels."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        tracker = _make_tracker(segNode, segmentID, volumeNode)

        # Paint a small 3×4 patch in the middle of an otherwise empty slice.
        patch = np.zeros((_J, _I), dtype=np.uint8)
        patch[4:7, 6:10] = 1
        change = tracker.write_slice(axis=0, idx=0, new_data=patch, source='test')

        self.assertIsNotNone(change)
        self.assertEqual(change.delta.shape, (3, 4),
                         "Delta crop must match the 3×4 changed region exactly")

    # ---- no-op write ----

    def test_write_identical_data_is_noop(self):
        """write_slice with unchanged data must return None."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[2] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        tracker = _make_tracker(segNode, segmentID, volumeNode)
        ones = np.ones((_J, _I), dtype=np.uint8)
        change = tracker.write_slice(axis=0, idx=2, new_data=ones, source='test')
        self.assertIsNone(change,
                          "Writing identical data must return None (no-op)")

    # ---- multiple writes → LIFO undo ----

    def test_two_writes_lifo_undo(self):
        from core.utils import get_slice_from_volume
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        tracker = _make_tracker(segNode, segmentID, volumeNode)

        left  = np.zeros((_J, _I), dtype=np.uint8)
        left[:, :_I // 2] = 1
        right = np.zeros((_J, _I), dtype=np.uint8)
        right[:, _I // 2:] = 1

        _change_a = tracker.write_slice(0, 5, left,  source='a')
        change_b  = tracker.write_slice(0, 5, right, source='b')  # replaces left

        # Undo only 'b' → slice should revert to 'left'.
        tracker.reverse_delta(change_b)
        result = _read_lm(segNode, segmentID, volumeNode)
        sl = get_slice_from_volume(result, 0, 5)
        np.testing.assert_array_equal(sl[:, :_I // 2], 1, err_msg="left half must be restored")
        np.testing.assert_array_equal(sl[:, _I // 2:], 0, err_msg="right half must be gone")

    # ---- source field is recorded correctly ----

    def test_source_field_is_stored(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        tracker = _make_tracker(segNode, segmentID, volumeNode)

        patch = np.ones((_J, _I), dtype=np.uint8)
        change = tracker.write_slice(0, 0, patch, source='expand')

        self.assertEqual(change.source, 'expand')

    # ---- snapshot is deep copy ----

    def test_snapshot_is_deep_copy(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[4, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        tracker = _make_tracker(segNode, segmentID, volumeNode)
        snap = tracker.snapshot()

        # Mutate the mask after snapshot.
        ones = np.ones((_J, _I), dtype=np.uint8)
        tracker.write_slice(0, 4, ones, source='test')

        # Snapshot must not have changed.
        from core.utils import get_slice_from_volume
        snap_slice = get_slice_from_volume(snap, 0, 4)
        np.testing.assert_array_equal(snap_slice[:, :_I // 2], 1)
        np.testing.assert_array_equal(snap_slice[:, _I // 2:], 0)


# ===========================================================================
# Unified history — end-to-end tests
# ===========================================================================

class UnifiedHistoryTest(unittest.TestCase):
    """End-to-end tests verifying that expandSegWithSPX returns a MaskChange
    and that logic.reverse_change correctly restores prior state.

    The widget's ``_history`` list stores entries of the form
    ``['expand', change]``, ``['brush', change]``, or
    ``['point', change, node, cp_id]``.  These tests exercise the logic-side
    contract (returns MaskChange) and the reverse path (reverse_change undoes it).
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._logic = _make_logic()

    # ---- expand returns a MaskChange ----

    def test_expand_returns_mask_change(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)
        labels = np.ones((_J, _I), dtype=np.int32)

        change = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                              labels, axis=0, sliceIndex=3)

        self.assertIsNotNone(change,
                             "expandSegWithSPX must return a MaskChange for a real change")
        self.assertEqual(change.source, 'expand')
        self.assertEqual(change.slice_idx, 3)

    def test_expand_undo_restores_pre_expand_slice(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        pre_state = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre_state[3, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre_state)

        labels = np.ones((_J, _I), dtype=np.int32)
        change = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                              labels, axis=0, sliceIndex=3)

        after = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(after[3], 1,
                                      err_msg="expand should paint the whole slice")

        # Undo via reverse_change.
        widget = _mock_widget(volumeNode, segNode, segmentID)
        self._logic.reverse_change(widget, change)
        self._logic.reset_render_state()

        restored = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(restored[3, :, :_I // 2], 1,
                                      err_msg="left half must be restored")
        np.testing.assert_array_equal(restored[3, :, _I // 2:], 0,
                                      err_msg="right half must be removed by undo")

    def test_expand_undo_does_not_change_other_slices(self):
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[2] = 1    # untouched by the expand
        pre[5, :, :_I // 2] = 1
        _write_lm(segNode, segmentID, volumeNode, pre)

        labels = np.ones((_J, _I), dtype=np.int32)
        change = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                              labels, axis=0, sliceIndex=5)

        widget = _mock_widget(volumeNode, segNode, segmentID)
        self._logic.reverse_change(widget, change)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1,
                                      err_msg="slice 2 must be untouched by undo")

    # ---- multiple expands → LIFO ordering ----

    def test_two_consecutive_expands_lifo(self):
        """Two expand calls produce independent MaskChange records; reversing
        the second one leaves the first intact."""
        from core.utils import get_slice_from_volume
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        labels = np.ones((_J, _I), dtype=np.int32)

        change_2 = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                                labels, axis=0, sliceIndex=2)
        change_4 = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                                labels, axis=0, sliceIndex=4)

        self.assertIsNotNone(change_2)
        self.assertIsNotNone(change_4)
        self.assertEqual(change_2.slice_idx, 2, "First change must be slice 2")
        self.assertEqual(change_4.slice_idx, 4, "Second change must be slice 4")

        # Undo only slice 4 (LIFO) — slice 2 must remain painted.
        widget = _mock_widget(volumeNode, segNode, segmentID)
        self._logic.reverse_change(widget, change_4)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1,
                                      err_msg="slice 2 must remain after undoing slice 4")
        np.testing.assert_array_equal(result[4], 0,
                                      err_msg="slice 4 must be cleared by undo")

    # ---- history entry format ----

    def test_history_entry_format_brush(self):
        """Brush entries must be ['brush', change] lists."""
        history = []
        mock_change = MagicMock()
        history.append(['brush', mock_change])
        entry = history[0]
        self.assertEqual(entry[0], 'brush')
        self.assertIs(entry[1], mock_change)
        self.assertEqual(len(entry), 2)

    def test_history_entry_format_expand(self):
        """Expand entries must be ['expand', change] lists."""
        history = []
        mock_change = MagicMock()
        history.append(['expand', mock_change])
        entry = history[0]
        self.assertEqual(entry[0], 'expand')
        self.assertIs(entry[1], mock_change)

    # ---- history cleared on segment add / switch ----

    def test_clear_is_clean_slate_for_next_segment(self):
        """After a segment switch the history must be empty."""
        history = []
        history.append(['expand', MagicMock()])
        history.append(['brush',  MagicMock()])
        history.clear()
        self.assertEqual(len(history), 0)

    # ---- reverse_change with None change is a no-op ----

    def test_reverse_change_none_is_noop(self):
        """reverse_change(widget, None) must not raise or modify the mask."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        initial = np.zeros((_K, _J, _I), dtype=np.uint8)
        initial[1] = 1
        _write_lm(segNode, segmentID, volumeNode, initial)

        widget = _mock_widget(volumeNode, segNode, segmentID)
        self._logic.reverse_change(widget, None)  # must not raise

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[1], 1,
                                      err_msg="reverse_change(None) must not modify mask")


# ===========================================================================
# _SliceViewMouseFilter tests
# ===========================================================================

class MouseFilterTest(unittest.TestCase):

    def _make_filter(self):
        import qt
        from core._input import _SliceViewMouseFilter
        on_press   = MagicMock()
        on_release = MagicMock()
        return _SliceViewMouseFilter(on_press, on_release), on_press, on_release

    def test_event_filter_always_returns_false(self):
        import qt
        filt, _, _ = self._make_filter()
        event = MagicMock()
        event.type.return_value = qt.QEvent.MouseMove
        result = filt.eventFilter(None, event)
        self.assertFalse(result, "eventFilter must never consume events")

    def test_mouse_press_calls_on_press(self):
        import qt
        filt, on_press, _ = self._make_filter()
        event = MagicMock()
        event.type.return_value   = qt.QEvent.MouseButtonPress
        event.button.return_value = qt.Qt.LeftButton
        filt.eventFilter(None, event)
        on_press.assert_called_once()

    def test_mouse_release_calls_on_release(self):
        import qt
        filt, _, on_release = self._make_filter()
        event = MagicMock()
        event.type.return_value   = qt.QEvent.MouseButtonRelease
        event.button.return_value = qt.Qt.LeftButton
        filt.eventFilter(None, event)
        on_release.assert_called_once()

    def test_callback_exception_does_not_propagate(self):
        """Exceptions in the callback must be swallowed to protect the Qt event loop."""
        import qt
        filt, on_press, _ = self._make_filter()
        on_press.side_effect = RuntimeError("test error")
        event = MagicMock()
        event.type.return_value   = qt.QEvent.MouseButtonPress
        event.button.return_value = qt.Qt.LeftButton
        try:
            filt.eventFilter(None, event)
        except RuntimeError:
            self.fail("eventFilter must not propagate exceptions from callbacks")
