"""Slicer-native integration tests for SegmentHumanBody.

Run from inside 3D Slicer via Developer Tools → Run Unittests, or via the
"Reload and Test" button, which delegates to SegmentHumanBodyTest.runTest().

These tests require a live Slicer process and exercise:
  - MRML scene operations (arrayFromSegmentBinaryLabelmap, etc.)
  - Delta-based undo: write_slice / reverse_delta round-trips.
  - Unified history: expand returns MaskChange; reverse_change restores state.
  - Qt event filter (_SliceViewMouseFilter): return value and on_press/on_release routing.
  - Segment creation handler lifecycle: onAddSegment cache/detach/create/restore flow,
    StrokeHandler.attach() supersession guard, full regression sequence.
  - Brush stroke workflow: commit_stroke, onUndo restores painted pixels, LIFO ordering.
  - Point placement workflow: _onPointConfirmed, positive/negative SPX selection, undo.
  - Manual point deletion: _onPointRemoved reverses the mask without Ctrl+Z.
  - Mixed-action undo: brush + point in one session, LIFO across action types.
  - Segment visibility: current segment (Q), saved/other segments (S checkbox),
    _apply_saved_segments_visibility isolation, multi-segment scenarios.
  - Window / Level: apply_window_level clipping and normalization, logic integration,
    W/L applied to model input before point placement and expansion.
  - CT-like workflow: brush delta on realistic intensity data, expandSegWithSPX on
    a real intensity gradient, multi-segment brush-then-undo across two segments.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import slicer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_K, _J, _I = 12, 15, 18   # volume shape used by all test classes

# Decorator that gates tests on the layout manager being available.
# Checked at test-run time (not module-import time) so that --python-script
# mode works even when the GUI finishes initializing after the module loads.
# Handles both method-level and class-level usage: when applied to a class the
# test methods inside are individually wrapped so the class stays a class and
# remains discoverable by unittest.TestLoader.
def _SKIP_LAYOUT(obj):
    import functools
    _MSG = "requires Slicer layout manager (run with full GUI, not --no-main-window)"
    def _guard(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            if slicer.app.layoutManager() is None:
                raise unittest.SkipTest(_MSG)
            return fn(*args, **kwargs)
        return _wrapper
    if isinstance(obj, type):
        for name in list(vars(obj)):
            if name.startswith('test') or name == 'setUp':
                setattr(obj, name, _guard(getattr(obj, name)))
        return obj
    return _guard(obj)


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


# ===========================================================================
# SegmentHumanBodyLogic tests
# ===========================================================================

class SegmentHumanBodyLogicTest(unittest.TestCase):
    """Integration tests for SegmentHumanBodyLogic."""

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._logic = _make_logic()

    def test_logic_reuses_tracker_for_same_segment(self):
        """Logic caches the tracker between consecutive operations on the same
        segment; a new instance is only created when segment or volume changes."""
        volumeNode = _make_volume()
        segNode, segmentID = _make_seg(volumeNode, with_segment=True)

        labels = np.ones((_J, _I), dtype=np.int32)
        seed = np.zeros((_K, _J, _I), dtype=np.uint8)
        seed[2, 0, 0] = 1
        seed[4, 0, 0] = 1
        _write_lm(segNode, segmentID, volumeNode, seed)

        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=2)
        tracker_id = id(self._logic._tracker)

        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=4)
        self.assertEqual(id(self._logic._tracker), tracker_id,
                         "Tracker must be reused across calls for the same segment")

    def test_remove_segment_selects_immediate_previous_segment(self):
        from SegmentHumanBody import SegmentHumanBodyWidget

        volumeNode = _make_volume()
        segNode, _ = _make_seg(volumeNode)
        segmentation = segNode.GetSegmentation()
        segA = segmentation.AddEmptySegment('SegA')
        segB = segmentation.AddEmptySegment('SegB')
        segC = segmentation.AddEmptySegment('SegC')

        self.assertEqual(
            SegmentHumanBodyWidget._segment_id_before(segNode, segC), segB)
        self.assertEqual(
            SegmentHumanBodyWidget._segment_id_before(segNode, segB), segA)

    def test_remove_first_segment_selects_next_segment(self):
        from SegmentHumanBody import SegmentHumanBodyWidget

        volumeNode = _make_volume()
        segNode, _ = _make_seg(volumeNode)
        segmentation = segNode.GetSegmentation()
        segA = segmentation.AddEmptySegment('SegA')
        segB = segmentation.AddEmptySegment('SegB')

        self.assertEqual(
            SegmentHumanBodyWidget._segment_id_before(segNode, segA), segB)

    def test_remove_only_segment_selects_none(self):
        from SegmentHumanBody import SegmentHumanBodyWidget

        volumeNode = _make_volume()
        segNode, segA = _make_seg(volumeNode, with_segment=True)

        self.assertEqual(
            SegmentHumanBodyWidget._segment_id_before(segNode, segA), '')

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

    @_SKIP_LAYOUT
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
        zeros = np.zeros((_J, _I), dtype=np.uint8)
        change = tracker.write_slice(axis=0, idx=3, new_data=zeros, source='test')
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

        # Seed one painted pixel so expand selects label 1 and produces a real change.
        seed = np.zeros((_K, _J, _I), dtype=np.uint8)
        seed[3, 0, 0] = 1
        _write_lm(segNode, segmentID, volumeNode, seed)

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

        # Seed one pixel per slice so each expand selects label 1 and produces a change.
        seed = np.zeros((_K, _J, _I), dtype=np.uint8)
        seed[2, 0, 0] = 1
        seed[4, 0, 0] = 1
        _write_lm(segNode, segmentID, volumeNode, seed)

        change_2 = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                                labels, axis=0, sliceIndex=2)
        change_4 = self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                                labels, axis=0, sliceIndex=4)

        self.assertIsNotNone(change_2)
        self.assertIsNotNone(change_4)
        self.assertEqual(change_2.slice_idx, 2, "First change must be slice 2")
        self.assertEqual(change_4.slice_idx, 4, "Second change must be slice 4")

        # Undo only slice 4 (LIFO) — slice 2 must remain fully painted.
        widget = _mock_widget(volumeNode, segNode, segmentID)
        self._logic.reverse_change(widget, change_4)

        result = _read_lm(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1,
                                      err_msg="slice 2 must remain after undoing slice 4")
        # Undo restores slice 4 to its pre-expand state: the single seed pixel at (0,0).
        expected_4 = np.zeros((_J, _I), dtype=np.uint8)
        expected_4[0, 0] = 1
        np.testing.assert_array_equal(result[4], expected_4,
                                      err_msg="slice 4 must be restored to pre-expand state")

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


# ===========================================================================
# Segment creation handler lifecycle — simulated user behaviour sequences
# ===========================================================================

class HandlerWrapperLifecycleTest(unittest.TestCase):
    """Regression tests for tool-handler mutual exclusion."""

    def setUp(self):
        import qt
        if not hasattr(qt, 'QObject'):
            raise unittest.SkipTest('requires a live Slicer Qt runtime')

    class _FakePlaceWidget:
        def __init__(self):
            self.placeModeEnabled = False
            self.place_multiple = None

        def setPlaceModeEnabled(self, active):
            self.placeModeEnabled = bool(active)

        def setPlaceMultipleMarkups(self, value):
            self.place_multiple = value

    class _FakeMarkupWidget:
        def __init__(self, place_widget, flip_on_set_current_node=False):
            self.place_widget = place_widget
            self.current_node = None
            self.flip_on_set_current_node = flip_on_set_current_node

        def findChildren(self, _klass):
            return [self.place_widget]

        def setCurrentNode(self, node):
            self.current_node = node
            if self.flip_on_set_current_node:
                self.place_widget.setPlaceModeEnabled(True)

    def _bind_prompt_helpers(self, widget):
        import types
        from SegmentHumanBody import SegmentHumanBodyWidget
        for name in (
            '_markup_place_widgets',
            '_prompt_place_states',
            '_set_prompt_place_states',
            '_set_prompt_widget_place_mode',
            '_deactivate_prompt_place_mode',
            '_set_prompt_nodes_preserving_place_mode',
            '_set_prompt_nodes',
        ):
            setattr(widget, name, types.MethodType(
                getattr(SegmentHumanBodyWidget, name), widget))

    def _make_prompt_widget_stub(self):
        w = MagicMock()
        w._suppressing_place_mode = False
        w.ui = MagicMock()
        w._pos_place = self._FakePlaceWidget()
        w._neg_place = self._FakePlaceWidget()
        w.ui.positivePrompts = self._FakeMarkupWidget(w._pos_place)
        w.ui.negativePrompts = self._FakeMarkupWidget(w._neg_place)
        self._bind_prompt_helpers(w)
        return w

    def _make_handler_widget_stub(self):
        w = self._make_prompt_widget_stub()
        vol = object()
        seg = MagicMock()
        seg.GetSegmentation.return_value.GetNumberOfSegments.return_value = 1
        w.logic = MagicMock()
        w.logic.getVolumeAndSegmentation.return_value = (vol, seg)
        w.ui.sourceVolumeSelector.currentNode.return_value = vol
        w.ui.segmentationNodeSelector.currentNode.return_value = seg
        w.ui.segmentSelector.currentSegmentID.return_value = 'Segment_1'
        w._parameterNode = MagicMock()
        w._active_handler = None
        w._active_prompt_widget = w.ui.positivePrompts
        w._ensure_current_prompt_nodes = MagicMock()
        return w

    def test_set_prompt_nodes_preserves_inactive_place_mode(self):
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._make_prompt_widget_stub()
        w.ui.positivePrompts.flip_on_set_current_node = True
        w.ui.negativePrompts.flip_on_set_current_node = True

        SegmentHumanBodyWidget._set_prompt_nodes(w, object(), object())

        self.assertFalse(w._pos_place.placeModeEnabled)
        self.assertFalse(w._neg_place.placeModeEnabled)

    def test_point_handler_attach_enables_wrapped_place_tool(self):
        from core._input import PointHandler

        w = self._make_handler_widget_stub()
        PointHandler().attach(w)

        self.assertIsInstance(w._active_handler, PointHandler)
        self.assertTrue(w._pos_place.placeModeEnabled)

    def test_point_handler_detach_disables_wrapped_place_tool(self):
        from core._input import PointHandler

        w = self._make_handler_widget_stub()
        handler = PointHandler()
        handler.attach(w)
        handler.detach(w)

        self.assertFalse(w._pos_place.placeModeEnabled)
        self.assertFalse(w._neg_place.placeModeEnabled)
        self.assertIsNone(w._active_handler)

    def test_stroke_attach_detaches_point_handler_and_disables_point_tool(self):
        from core._input import BrushHandler, PointHandler

        w = self._make_handler_widget_stub()
        PointHandler().attach(w)
        self.assertTrue(w._pos_place.placeModeEnabled)

        original_on_attach = BrushHandler._on_attach
        try:
            BrushHandler._on_attach = lambda self_h, widget: None
            BrushHandler().attach(w)
        finally:
            BrushHandler._on_attach = original_on_attach

        self.assertFalse(w._pos_place.placeModeEnabled)
        self.assertFalse(w._neg_place.placeModeEnabled)
        self.assertIsInstance(w._active_handler, BrushHandler)


class AddSegmentHandlerTest(unittest.TestCase):
    """Slicer-native behaviour tests for handler lifecycle around segment creation.

    Each test simulates an actual user action sequence end-to-end:
      - real MRML volume + segmentation nodes
      - real Slicer interaction node (placement mode transitions)
      - Qt signal chain replicated via side_effect so onSegmentChanged →
        clearPrompts fires exactly as it does in the live module

    Regression: without the fix, clearPrompts() set _active_handler =
    PointHandler() directly, orphaning the active StrokeHandler and breaking
    the "click place button to deactivate brush" flow.
    """

    # ------------------------------------------------------------------
    # Set-up / tear-down
    # ------------------------------------------------------------------

    def setUp(self):
        slicer.mrmlScene.Clear()
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._paramNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._paramNode)
        self._volNode = _make_volume()
        self._segNode, _ = _make_seg(self._volNode)   # 0 segments initially
        self._logic.setVolumeAndSegmentation(self._paramNode, self._volNode, self._segNode)
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    # ------------------------------------------------------------------
    # Widget stub
    # ------------------------------------------------------------------

    def _make_widget(self):
        """Minimal duck-typed widget that lets onAddSegment + clearPrompts run
        with real MRML nodes while keeping Slicer UI widgets mocked.

        The mock segmentSelector.setCurrentSegmentID is wired via side_effect
        to call onSegmentChanged directly, replicating the Qt signal chain that
        fires in the live module without needing a loaded Qt widget.
        """
        from core._state import WidgetState
        from SegmentHumanBody import SegmentHumanBodyWidget

        logic     = self._logic
        paramNode = self._paramNode
        volNode   = self._volNode
        segNode   = self._segNode

        class _W:
            def _segEditor(self):              return None
            def _observeMarkupsNodes(self):    pass
            def _applyBrushParams(self):       pass
            def _get_composite_node(self, *a): return None
            def getOrCreateSegmentationNode(self):
                return self.ui.segmentationNodeSelector.currentNode()
            def onAddSegment(self, *args):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget.onAddSegment(self, *args)
            def clearPrompts(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget.clearPrompts(self)
            def _restore_prompt_placement(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._restore_prompt_placement(self)
            def _apply_saved_segments_visibility(self, exclude=None):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._apply_saved_segments_visibility(self, exclude=exclude)
            def _sync_annotation_visibility(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._sync_annotation_visibility(self)
            def _clear_history(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._clear_history(self)

        w = _W()
        w.ctrl                     = WidgetState(w)
        w._active_handler          = None
        w._history                 = []
        w._redo_stack              = []
        w.logic                    = logic
        w._parameterNode           = paramNode
        w._spx_boundary_visible    = False
        w._spx_boundary_node       = None
        w._saved_segments_visible  = False
        w._current_segment_visible = True
        w._acknowledged_segment_id = None
        w.modelFamily              = None

        w.ui = MagicMock()
        w.ui.sourceVolumeSelector.currentNode.return_value      = volNode
        w.ui.segmentationNodeSelector.currentNode.return_value  = segNode
        w.ui.segmentSelector.currentNode.return_value           = segNode
        w.ui.segmentSelector.currentSegmentID.return_value      = ''
        w.ui.brushToolButton.isChecked.return_value  = False
        w.ui.eraseToolButton.isChecked.return_value  = False

        # Replicate Qt signal: setCurrentSegmentID → onSegmentChanged → clearPrompts
        def _on_set_segment(segID):
            SegmentHumanBodyWidget.onSegmentChanged(w, segID)

        w.ui.segmentSelector.setCurrentSegmentID.side_effect = _on_set_segment

        return w

    # ------------------------------------------------------------------
    # Sequence 1: "add segment while brush active → brush must be restored"
    # ------------------------------------------------------------------

    def test_add_segment_while_brush_active_restores_brush(self):
        """User sequence: brush active → Add Segment button → brush still active.

        Without the fix: clearPrompts (called via onSegmentChanged) sets
        _active_handler = PointHandler() directly, orphaning the brush.
        With the fix: onAddSegment caches, detaches, creates, then re-attaches.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import BrushHandler

        w = self._make_widget()
        w._active_handler = BrushHandler()

        SegmentHumanBodyWidget.onAddSegment(w)

        self.assertIsInstance(w._active_handler, BrushHandler,
            "BrushHandler must be restored as active handler after onAddSegment")

    def test_add_segment_while_erase_active_restores_erase(self):
        """Same contract for EraseHandler."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import EraseHandler

        w = self._make_widget()
        w._active_handler = EraseHandler()

        SegmentHumanBodyWidget.onAddSegment(w)

        self.assertIsInstance(w._active_handler, EraseHandler)

    def test_add_segment_without_stroke_handler_stays_in_point_mode(self):
        """Without a prior stroke handler, onAddSegment leaves point-placement
        mode active (PointHandler set by clearPrompts, nothing to restore)."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import PointHandler

        w = self._make_widget()
        # _active_handler is None — no stroke handler was active.

        SegmentHumanBodyWidget.onAddSegment(w)

        self.assertIsInstance(w._active_handler, PointHandler)

    # ------------------------------------------------------------------
    # Sequence 2: detach order — brush must be gone before creation fires
    # ------------------------------------------------------------------

    def test_brush_is_detached_before_segment_creation(self):
        """onAddSegment must detach the stroke handler BEFORE the segment is
        created so clearPrompts never encounters a live StrokeHandler.

        Verified by intercepting the setCurrentSegmentID side_effect (which
        fires right after AddEmptySegment) and recording handler state there.
        At that moment _active_handler must already be None (detached), not
        BrushHandler.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import BrushHandler

        w = self._make_widget()
        w._active_handler = BrushHandler()

        active_at_creation = []

        def _on_set_segment(segID):
            # Fires immediately after AddEmptySegment, before clearPrompts.
            active_at_creation.append(type(w._active_handler).__name__)
            SegmentHumanBodyWidget.onSegmentChanged(w, segID)

        w.ui.segmentSelector.setCurrentSegmentID.side_effect = _on_set_segment

        SegmentHumanBodyWidget.onAddSegment(w)

        self.assertEqual(len(active_at_creation), 1)
        self.assertEqual(active_at_creation[0], 'NoneType',
            "_active_handler must be None (detached) when segment creation "
            "triggers onSegmentChanged — not an orphaned BrushHandler")

    # ------------------------------------------------------------------
    # Sequence 3: full regression — brush → add segment → click place
    # ------------------------------------------------------------------

    def test_brush_active_add_segment_then_place_deactivates_brush(self):
        """Full regression sequence:
          1. Brush active
          2. Add Segment (triggers clearPrompts internally)
          3. Click positive-prompts place button (_onPlaceModeChanged)
          → place click must deactivate brush and activate PointHandler.

        This is the exact sequence that failed before the fix.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import BrushHandler, PointHandler

        w = self._make_widget()
        w._active_handler = BrushHandler()   # step 1: brush is active

        SegmentHumanBodyWidget.onAddSegment(w)                 # step 2
        self.assertIsInstance(w._active_handler, BrushHandler,
            "Brush must still be active immediately after Add Segment")

        SegmentHumanBodyWidget._onPlaceModeChanged(w, active=True)  # step 3

        self.assertIsInstance(w._active_handler, PointHandler,
            "Place button click must deactivate brush and activate PointHandler")

    # ------------------------------------------------------------------
    # Sequence 4: StrokeHandler.attach() supersession guard
    # ------------------------------------------------------------------

    def test_stroke_handler_attach_bails_out_when_superseded_by_add_segment(self):
        """If _activate_effect triggers onAddSegment which replaces _active_handler
        via the cache/restore path, the original StrokeHandler.attach() must bail
        out at the supersession guard and not register itself as the active handler.

        Simulated by patching _activate_effect on the class to call onAddSegment
        directly (replicating the 0-segment path that fires in production).
        """
        from core._input import BrushHandler
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._make_widget()
        original_handler = BrushHandler()

        original_activate = BrushHandler._activate_effect

        _called = [False]
        def _hijack_activate(self_h, widget):
            # Simulate: Paint activated on 0-segment segmentation → onAddSegment.
            # One-shot guard: the restore path inside onAddSegment creates a fresh
            # BrushHandler and calls attach() on it, which would re-enter here and
            # recurse infinitely without this check.
            if _called[0]:
                return
            _called[0] = True
            SegmentHumanBodyWidget.onAddSegment(widget)

        BrushHandler._activate_effect = _hijack_activate
        try:
            original_handler.attach(w)
        finally:
            BrushHandler._activate_effect = original_activate

        # The resumed handler (created inside onAddSegment's finally) must be active,
        # not the original instance that was superseded.
        self.assertIsNot(w._active_handler, original_handler,
            "The superseded original handler must not end up as _active_handler")
        self.assertIsInstance(w._active_handler, BrushHandler,
            "A fresh BrushHandler from the resume path must be active")

    # ------------------------------------------------------------------
    # Sequence 5: unified pre-attach segment guard
    # ------------------------------------------------------------------

    def test_point_handler_attach_creates_segment_when_empty(self):
        """PointHandler.attach() on a 0-segment segmentation auto-creates a
        segment before activating placement mode.

        Bug regression: previously PointHandler had no segment check and went
        straight to Slicer placement mode.  The segment was only created lazily
        on the first click (inside commit_point → _ensure_seg_and_segment),
        which caused a deferred qMRMLSegmentSelectorWidget signal to fire and
        call clearPrompts(), detaching the handler mid-session.

        Fix: _detach_current_tool_if_exists (base class, called by all handlers)
        calls onAddSegment() when no segment exists — before any Slicer
        interaction element is installed.
        """
        from core._input import PointHandler

        w = self._make_widget()
        segNode = w.ui.segmentationNodeSelector.currentNode()
        self.assertEqual(segNode.GetSegmentation().GetNumberOfSegments(), 0,
            "Pre-condition: segmentation starts with 0 segments")

        ph = PointHandler()
        try:
            ph.attach(w)
            self.assertEqual(
                segNode.GetSegmentation().GetNumberOfSegments(), 1,
                "PointHandler.attach() must create a segment before placement mode")
            self.assertIsInstance(w._active_handler, PointHandler,
                "PointHandler must remain active after attach on empty segmentation")
        finally:
            ph.detach(w)

    def test_erase_handler_attach_creates_segment_when_empty(self):
        """EraseHandler.attach() on a 0-segment segmentation auto-creates a
        segment before activating the Erase effect.

        The old per-handler check in _activate_effect only guarded the 'Paint'
        effect (self.EFFECT == 'Paint'); 'Erase' was never covered.  The
        unified guard in _detach_current_tool_if_exists covers all handlers.
        """
        from core._input import EraseHandler

        w = self._make_widget()
        segNode = w.ui.segmentationNodeSelector.currentNode()
        self.assertEqual(segNode.GetSegmentation().GetNumberOfSegments(), 0,
            "Pre-condition: segmentation starts with 0 segments")

        eh = EraseHandler()
        try:
            eh.attach(w)
            self.assertEqual(
                segNode.GetSegmentation().GetNumberOfSegments(), 1,
                "EraseHandler.attach() must create a segment before Erase effect")
            self.assertIsInstance(w._active_handler, EraseHandler,
                "EraseHandler must remain active after attach on empty segmentation")
        finally:
            eh.detach(w)

    def test_brush_handler_attach_creates_segment_when_empty(self):
        """BrushHandler.attach() auto-creates a segment via the unified guard.

        Verifies that the guard in _detach_current_tool_if_exists fires for
        BrushHandler and that _active_handler ends up pointing to the
        BrushHandler (not superseded by the onAddSegment restore path, since no
        prior stroke handler was active when attach() was called).
        """
        from core._input import BrushHandler

        w = self._make_widget()
        segNode = w.ui.segmentationNodeSelector.currentNode()
        self.assertEqual(segNode.GetSegmentation().GetNumberOfSegments(), 0,
            "Pre-condition: segmentation starts with 0 segments")

        bh = BrushHandler()
        try:
            bh.attach(w)
            self.assertEqual(
                segNode.GetSegmentation().GetNumberOfSegments(), 1,
                "BrushHandler.attach() must create a segment before Paint effect")
            self.assertIsInstance(w._active_handler, BrushHandler,
                "BrushHandler must remain active after attach on empty segmentation")
        finally:
            bh.detach(w)


# ===========================================================================
# Shared helpers for workflow tests
# ===========================================================================

def _setup_red_slice_at_k(volNode, k):
    """Position the Red (axial) slice view at voxel K-index *k*.

    Also ensures a 4-view layout so the Red slice widget is present.
    Returns the IJK→RAS vtkMatrix4x4 so callers can compute RAS
    coordinates for control-point placement on this slice.
    """
    import vtk
    slicer.app.layoutManager().setLayout(
        slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    ijkToRAS = vtk.vtkMatrix4x4()
    volNode.GetIJKToRASMatrix(ijkToRAS)
    ras_origin = ijkToRAS.MultiplyPoint([0.0, 0.0, float(k), 1.0])
    slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode().SetSliceOffset(
        ras_origin[2])
    return ijkToRAS


def _ijk_to_ras_pt(ijkToRAS, col, row, k):
    """Convert IJK voxel coordinates to an [x, y, z] RAS list."""
    ras = ijkToRAS.MultiplyPoint([float(col), float(row), float(k), 1.0])
    return [ras[0], ras[1], ras[2]]


def _make_spx_family(labels):
    """Return a minimal SPX family stub whose on_expand always returns *labels*.

    ``model`` is a truthy sentinel so the ``modelFamily.model`` guard in
    ``commit_point`` / ``on_expand`` passes without loading a real model.
    """
    class _FakeSPXFamily:
        model = object()
        def on_expand(self, img, **kwargs):
            return labels
    return _FakeSPXFamily()


def _make_undo_test_widget(volNode, segNode, segID, paramNode, logic,
                           modelFamily=None):
    """Duck-typed widget stub for brush / point / undo workflow tests.

    Wires the segmentSelector.setCurrentSegmentID side_effect to replicate
    the Qt signal chain (setCurrentSegmentID → onSegmentChanged → clearPrompts)
    that fires in the live module, so the stub exercises the same code paths.

    Parameters
    ----------
    volNode, segNode, segID, paramNode : real MRML nodes
    logic : SegmentHumanBodyLogic instance
    modelFamily : family stub for point-placement tests; None for brush tests
    """
    from core._state import WidgetState
    from SegmentHumanBody import SegmentHumanBodyWidget

    class _W:
        def _segEditor(self):              return None
        def _observeMarkupsNodes(self):    pass
        def _applyBrushParams(self):       pass
        def _get_composite_node(self, *a): return None
        def _resolveActiveView(self):      pass
        def getUserParameters(self):       return {}
        def getOrCreateSegmentationNode(self):
            return self.ui.segmentationNodeSelector.currentNode()
        def clearPrompts(self):
            from SegmentHumanBody import SegmentHumanBodyWidget
            SegmentHumanBodyWidget.clearPrompts(self)
        def _apply_saved_segments_visibility(self, exclude=None):
            from SegmentHumanBody import SegmentHumanBodyWidget
            SegmentHumanBodyWidget._apply_saved_segments_visibility(self, exclude=exclude)
        def _clear_history(self):
            from SegmentHumanBody import SegmentHumanBodyWidget
            SegmentHumanBodyWidget._clear_history(self)
        def _restore_prompt_placement(self):
            from SegmentHumanBody import SegmentHumanBodyWidget
            SegmentHumanBodyWidget._restore_prompt_placement(self)
        def _sync_annotation_visibility(self):
            from SegmentHumanBody import SegmentHumanBodyWidget
            SegmentHumanBodyWidget._sync_annotation_visibility(self)

    w = _W()
    w.ctrl                     = WidgetState(w)
    w._active_handler          = None
    w._history                 = []
    w._redo_stack              = []
    w.logic                    = logic
    w._parameterNode           = paramNode
    w.currentViewName          = "Red"
    w.modelFamily              = modelFamily
    w._spx_boundary_visible    = False
    w._spx_boundary_node       = None
    w._saved_segments_visible  = False
    w._current_segment_visible = True
    w._acknowledged_segment_id = segID   # segment already selected; suppress deferred duplicate

    w.ui = MagicMock()
    w.ui.sourceVolumeSelector.currentNode.return_value        = volNode
    w.ui.segmentationNodeSelector.currentNode.return_value    = segNode
    w.ui.segmentSelector.currentNode.return_value             = segNode
    w.ui.segmentSelector.currentSegmentID.return_value        = segID
    w.ui.brushToolButton.isChecked.return_value               = False
    w.ui.eraseToolButton.isChecked.return_value               = False

    def _on_set_segment(sid):
        SegmentHumanBodyWidget.onSegmentChanged(w, sid)

    w.ui.segmentSelector.setCurrentSegmentID.side_effect = _on_set_segment
    return w


# ===========================================================================
# Brush stroke commit and undo
# ===========================================================================

class BrushStrokeUndoTest(unittest.TestCase):
    """Simulates the user selecting the brush tool, painting strokes, then
    pressing Ctrl+Z to undo them.

    ``commit_stroke`` is called directly (bypassing the real Segment Editor
    effect, which requires a visible Qt window) by writing the desired
    after-state directly into Slicer's labelmap — the same data that the
    Segment Editor Paint effect would produce.  ``onUndo`` is then called
    as the user would trigger it via Ctrl+Z.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)

    def tearDown(self):
        slicer.mrmlScene.Clear()

    def _w(self):
        return _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode, self._logic)

    # ------------------------------------------------------------------
    # Helper: simulate one brush stroke on *axis/idx* by writing the
    # desired after-state into Slicer, then calling commit_stroke.
    # ------------------------------------------------------------------

    def _do_stroke(self, w, axis, idx, before_2d, after_2d, source='brush'):
        """Write *after_2d* to Slicer then commit the stroke delta."""
        full = np.zeros((_K, _J, _I), dtype=np.uint8)
        # Preserve any existing data already in Slicer for other slices.
        existing = _read_lm(self._segNode, self._segID, self._volNode)
        full[:] = existing
        # Overwrite just the target slice with the after-state.
        from core.utils import write_slice_to_volume
        write_slice_to_volume(full, after_2d, axis, idx)
        _write_lm(self._segNode, self._segID, self._volNode, full)
        return self._logic.commit_stroke(w, axis, idx, before_2d, source=source)

    # ------------------------------------------------------------------

    def test_brush_stroke_records_history_entry(self):
        """Painting a stroke → commit_stroke returns a MaskChange that callers
        push to _history as ['brush', change]."""
        w = self._w()
        before = np.zeros((_J, _I), dtype=np.uint8)
        after  = np.zeros((_J, _I), dtype=np.uint8)
        after[3:8, 4:12] = 1

        change = self._do_stroke(w, axis=0, idx=5, before_2d=before, after_2d=after)

        self.assertIsNotNone(change,
            "commit_stroke must return a MaskChange for a real paint stroke")
        self.assertEqual(change.source, 'brush')
        self.assertTrue(np.any(change.delta > 0),
            "Brush delta must contain positive values (pixels added)")

    def test_brush_undo_restores_empty_slice(self):
        """User workflow: brush paints slice 5 → Ctrl+Z → slice 5 empty again."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        before = np.zeros((_J, _I), dtype=np.uint8)
        after  = np.zeros((_J, _I), dtype=np.uint8)
        after[3:8, 4:12] = 1

        change = self._do_stroke(w, axis=0, idx=5, before_2d=before, after_2d=after)
        w._history.append(['brush', change])

        SegmentHumanBodyWidget.onUndo(w)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[5], 0,
            "Brush undo must restore the painted slice to empty")

    def test_brush_undo_leaves_other_slices_untouched(self):
        """Undoing a stroke on slice 5 must not affect a previously painted slice 3."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()

        # Slice 3 is already painted from a previous session.
        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[3] = 1
        _write_lm(self._segNode, self._segID, self._volNode, pre)

        # Stroke on slice 5.
        before_5 = np.zeros((_J, _I), dtype=np.uint8)
        after_5  = np.zeros((_J, _I), dtype=np.uint8)
        after_5[2:7, 2:10] = 1

        change = self._do_stroke(w, axis=0, idx=5, before_2d=before_5, after_2d=after_5)
        w._history.append(['brush', change])

        SegmentHumanBodyWidget.onUndo(w)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[3], 1,
            "Slice 3 must be untouched after undoing a stroke on slice 5")
        np.testing.assert_array_equal(result[5], 0,
            "Stroke on slice 5 must be undone")

    def test_two_strokes_undo_lifo_order(self):
        """Two brush strokes on different slices; Ctrl+Z twice pops LIFO."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()

        # Stroke A: left half of slice 4.
        before_a = np.zeros((_J, _I), dtype=np.uint8)
        after_a  = np.zeros((_J, _I), dtype=np.uint8)
        after_a[:, :_I // 2] = 1
        change_a = self._do_stroke(w, axis=0, idx=4, before_2d=before_a, after_2d=after_a)
        w._history.append(['brush', change_a])

        # Stroke B: right half of slice 7 (slice 4 is already painted).
        before_b = np.zeros((_J, _I), dtype=np.uint8)
        after_b  = np.zeros((_J, _I), dtype=np.uint8)
        after_b[:, _I // 2:] = 1
        change_b = self._do_stroke(w, axis=0, idx=7, before_2d=before_b, after_2d=after_b)
        w._history.append(['brush', change_b])

        # First Ctrl+Z: stroke B (the most recent) is undone.
        SegmentHumanBodyWidget.onUndo(w)
        r1 = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(r1[4, :, :_I // 2], 1,
            "Stroke A must remain after undoing stroke B")
        np.testing.assert_array_equal(r1[7], 0,
            "Stroke B (slice 7) must be cleared by the first undo")

        # Second Ctrl+Z: stroke A is undone.
        SegmentHumanBodyWidget.onUndo(w)
        r2 = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(r2[4], 0,
            "Stroke A (slice 4) must be cleared by the second undo")

    def test_erase_removing_pixels_is_tracked(self):
        """Erasing pixels that exist produces a negative-delta entry in history."""
        from core._input import EraseHandler

        w = self._w()

        # First paint the full slice 5.
        before_paint = np.zeros((_J, _I), dtype=np.uint8)
        after_paint  = np.ones((_J, _I), dtype=np.uint8)
        paint_change = self._do_stroke(w, axis=0, idx=5,
                                       before_2d=before_paint, after_2d=after_paint,
                                       source='brush')
        w._history.append(['brush', paint_change])

        # Erase the left half: before = full row; after = right half only.
        before_erase = np.ones((_J, _I), dtype=np.uint8)
        after_erase  = np.zeros((_J, _I), dtype=np.uint8)
        after_erase[:, _I // 2:] = 1
        erase_change = self._do_stroke(w, axis=0, idx=5,
                                       before_2d=before_erase, after_2d=after_erase,
                                       source='erase')

        handler = EraseHandler()
        self.assertIsNotNone(erase_change,
            "Erasing painted pixels must produce a MaskChange")
        self.assertTrue(handler._should_track(erase_change),
            "EraseHandler._should_track must return True when pixels were removed")
        self.assertTrue(np.any(erase_change.delta < 0),
            "Erase delta must contain negative values for removed pixels")

    def test_erase_noop_not_tracked(self):
        """Erasing over empty pixels produces no change → _should_track returns False."""
        from core._input import EraseHandler

        w = self._w()

        # Slice 5 is empty; before and after are identical (erasing air).
        before = np.zeros((_J, _I), dtype=np.uint8)
        after  = np.zeros((_J, _I), dtype=np.uint8)
        change = self._do_stroke(w, axis=0, idx=5,
                                 before_2d=before, after_2d=after, source='erase')

        # commit_stroke returns None for a no-op — nothing to track.
        self.assertIsNone(change,
            "Erasing empty pixels must return None from commit_stroke (no-op)")


# ===========================================================================
# Point placement (positive and negative prompts) and undo
# ===========================================================================

@_SKIP_LAYOUT
class PointPlacementUndoTest(unittest.TestCase):
    """Simulates clicking positive or negative prompt points on a slice and
    undoing them via Ctrl+Z.

    A synthetic SPX label map (left-half = label 1, right-half = label 2) is
    injected so tests are deterministic without running a real SPX algorithm.
    The Red slice is positioned at axial index *_TARGET_K* so control points
    placed there are on the same slice the model sees.
    """

    _TARGET_K = 5    # axial slice index the tests operate on

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        self._ijkToRAS = _setup_red_slice_at_k(self._volNode, self._TARGET_K)

        # Synthetic SPX labels: left half = 1, right half = 2.
        synthetic = np.ones((_J, _I), dtype=np.int32)
        synthetic[:, _I // 2:] = 2
        self._spx_fam  = _make_spx_family(synthetic)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def _w(self):
        return _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode,
            self._logic, modelFamily=self._spx_fam)

    def _place_pos_point(self, col, row):
        """Add a positive control point at IJK (col, row, _TARGET_K) and
        fire _onPointConfirmed.  Returns the widget used."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w()
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, col, row, self._TARGET_K)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)
        return w, posNode

    # ------------------------------------------------------------------

    def test_positive_point_adds_left_spx_region(self):
        """Clicking col=4 (left half, SPX label 1) must paint the entire left
        half of the axial slice while leaving the right half untouched."""
        w, _ = self._place_pos_point(col=4, row=3)
        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._TARGET_K, :, :_I // 2], 1,
            "Positive point in left half must paint the left SPX region")
        np.testing.assert_array_equal(result[self._TARGET_K, :, _I // 2:], 0,
            "Right half (SPX label 2) must not be painted by a left-half click")

    def test_positive_point_right_half_adds_right_spx_region(self):
        """Clicking col=11 (right half, SPX label 2) must paint the right half."""
        w, _ = self._place_pos_point(col=11, row=3)
        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._TARGET_K, :, _I // 2:], 1,
            "Positive point in right half must paint the right SPX region")
        np.testing.assert_array_equal(result[self._TARGET_K, :, :_I // 2], 0,
            "Left half must not be painted by a right-half click")

    def test_negative_point_subtracts_spx_region(self):
        """Clicking a negative point removes the matching SPX region from
        an already-painted mask, leaving the rest intact."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        # Pre-paint the entire slice.
        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[self._TARGET_K] = 1
        _write_lm(self._segNode, self._segID, self._volNode, pre)

        w = self._w()
        _, negNode = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, self._TARGET_K)
        negNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, negNode)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._TARGET_K, :, :_I // 2], 0,
            "Negative point in left half must subtract the left SPX region")
        np.testing.assert_array_equal(result[self._TARGET_K, :, _I // 2:], 1,
            "Right half was not touched by the negative point — must remain")

    def test_undo_removes_control_point_and_restores_mask(self):
        """Ctrl+Z after a positive-point placement must:
          1. Remove the control point from the markup node (or recreate the node empty).
          2. Restore the mask to its state before the click.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w, _ = self._place_pos_point(col=4, row=3)

        after = _read_lm(self._segNode, self._segID, self._volNode)
        self.assertTrue(np.any(after[self._TARGET_K] > 0),
            "Mask must have changed after positive point placement")

        SegmentHumanBodyWidget.onUndo(w)

        # After undo the node may have been recreated (counter reset); in either
        # case the current posNode must have zero confirmed control points.
        posNode_after, _ = self._logic.getPromptNodes(self._pNode)
        self.assertEqual(posNode_after.GetNumberOfControlPoints(), 0,
            "Positive prompt node must have no points after undo")

        restored = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(restored[self._TARGET_K], 0,
            "Mask must be fully empty after undoing the only positive point")

    def test_two_points_undo_lifo_order(self):
        """Place positive point A (left half) then B (right half).
        First Ctrl+Z removes B's pixels while A remains; second removes A.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        posNode, _ = self._logic.getPromptNodes(self._pNode)

        # Point A — left half.
        ras_a = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, self._TARGET_K)
        posNode.AddControlPoint(ras_a)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)

        # Point B — right half (unions with the existing left-half paint).
        ras_b = _ijk_to_ras_pt(self._ijkToRAS, 11, 3, self._TARGET_K)
        posNode.AddControlPoint(ras_b)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)

        after_both = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(after_both[self._TARGET_K], 1,
            "Both SPX halves must be painted after two positive points")

        # First undo: point B removed, right half gone.
        SegmentHumanBodyWidget.onUndo(w)
        r1 = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(r1[self._TARGET_K, :, :_I // 2], 1,
            "Left half must remain after undoing point B")
        np.testing.assert_array_equal(r1[self._TARGET_K, :, _I // 2:], 0,
            "Right half must be cleared after undoing point B")

        # Second undo: point A removed, mask empty.
        SegmentHumanBodyWidget.onUndo(w)
        r2 = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(r2[self._TARGET_K], 0,
            "Mask must be fully empty after undoing both points")

    def test_point_on_other_slice_is_silently_ignored(self):
        """A control point placed on a slice different from the current Red view
        must produce no mask change (ras_to_ijk filters it out)."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        posNode, _ = self._logic.getPromptNodes(self._pNode)

        # Place the point at K=9 (different from _TARGET_K=5).
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, 9)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._TARGET_K], 0,
            "Mask must be untouched when the point is on a different slice")
        # History entry is still pushed (with change=None) because the widget
        # always records the event — only the mask state confirms it was a no-op.
        self.assertEqual(len(w._history), 1)
        self.assertEqual(w._history[0][0], 'point')
        self.assertIsNone(w._history[0][1],
            "MaskChange must be None when the point was off-slice")


# ===========================================================================
# Manual point deletion (_onPointRemoved)
# ===========================================================================

@_SKIP_LAYOUT
class ManualPointDeletionTest(unittest.TestCase):
    """Simulates the user placing a prompt point then deleting it directly
    through the Markups module or the control-point list — without using Ctrl+Z.

    In the live module a VTK PointRemovedEvent fires and _onPointRemoved
    reverses the mask entry.  In tests we call _onPointRemoved directly
    after removing the point from the node.
    """

    _TARGET_K = 6

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        self._ijkToRAS = _setup_red_slice_at_k(self._volNode, self._TARGET_K)
        synthetic = np.ones((_J, _I), dtype=np.int32)
        synthetic[:, _I // 2:] = 2
        self._spx_fam = _make_spx_family(synthetic)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def _w(self):
        return _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode,
            self._logic, modelFamily=self._spx_fam)

    def _place_and_confirm(self, w, col, row):
        """Place a positive point at (col, row, _TARGET_K) and fire
        _onPointConfirmed.  Returns the markup node and the control-point ID."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, col, row, self._TARGET_K)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)
        cp_id = posNode.GetNthControlPointID(posNode.GetNumberOfControlPoints() - 1)
        return posNode, cp_id

    # ------------------------------------------------------------------

    def test_manual_delete_reverses_mask_change(self):
        """User places a positive point (left half painted) then manually
        deletes the control point.  _onPointRemoved must restore the mask."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        posNode, cp_id = self._place_and_confirm(w, col=4, row=3)

        after = _read_lm(self._segNode, self._segID, self._volNode)
        self.assertTrue(np.any(after[self._TARGET_K] > 0),
            "Mask must be non-empty after point placement")

        # Simulate the user deleting the control point from the Markups list.
        idx = posNode.GetControlPointIndexByID(cp_id)
        posNode.RemoveNthControlPoint(idx)

        # In the live module a VTK event fires _onPointRemoved automatically;
        # in tests we call it directly with the node as caller.
        SegmentHumanBodyWidget._onPointRemoved(w, posNode)

        restored = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(restored[self._TARGET_K], 0,
            "Manually deleting the control point must reverse its mask change")

    def test_manual_delete_removes_history_entry(self):
        """After _onPointRemoved the history entry for that point must be gone."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        posNode, cp_id = self._place_and_confirm(w, col=4, row=3)

        self.assertEqual(len(w._history), 1)
        self.assertEqual(w._history[0][0], 'point')

        idx = posNode.GetControlPointIndexByID(cp_id)
        posNode.RemoveNthControlPoint(idx)
        SegmentHumanBodyWidget._onPointRemoved(w, posNode)

        self.assertEqual(len(w._history), 0,
            "_onPointRemoved must remove the matching history entry")

    def test_manual_delete_while_paused_is_suppressed(self):
        """_onPointRemoved is a no-op when ctrl.is_paused (e.g. during undo or
        clearPrompts) so it does not double-apply the reversal."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()
        posNode, cp_id = self._place_and_confirm(w, col=4, row=3)

        after = _read_lm(self._segNode, self._segID, self._volNode).copy()

        # Pause and manually delete — _onPointRemoved must be a no-op.
        idx = posNode.GetControlPointIndexByID(cp_id)
        posNode.RemoveNthControlPoint(idx)
        w.ctrl.pause()
        try:
            SegmentHumanBodyWidget._onPointRemoved(w, posNode)
        finally:
            w.ctrl.resume()

        # Mask and history must be unchanged.
        still = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(still[self._TARGET_K],
                                      after[self._TARGET_K],
            "Mask must not change when _onPointRemoved fires while paused")
        self.assertEqual(len(w._history), 1,
            "History entry must not be removed when _onPointRemoved fires while paused")


# ===========================================================================
# Mixed-action undo (brush + point + expand in one session)
# ===========================================================================

@_SKIP_LAYOUT
class MixedActionUndoTest(unittest.TestCase):
    """Simulates a realistic annotation session that combines brush strokes
    and point placements, then undoes them in LIFO order.

    This is the primary regression test for the unified _history stack: each
    undo must pop exactly its own action regardless of the types of other
    entries in the stack.
    """

    _BRUSH_K = 4    # axial slice index used for the brush stroke
    _POINT_K = 5    # axial slice index used for the point placement

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        # Red slice starts at the point's slice; tests move it as needed.
        self._ijkToRAS = _setup_red_slice_at_k(self._volNode, self._POINT_K)
        synthetic = np.ones((_J, _I), dtype=np.int32)
        synthetic[:, _I // 2:] = 2
        self._spx_fam = _make_spx_family(synthetic)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def _w(self):
        return _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode,
            self._logic, modelFamily=self._spx_fam)

    def _do_brush_stroke(self, w, k, after_2d, source='brush'):
        from core.utils import write_slice_to_volume
        full = _read_lm(self._segNode, self._segID, self._volNode).copy()
        before_2d = full[k].copy()
        write_slice_to_volume(full, after_2d, 0, k)
        _write_lm(self._segNode, self._segID, self._volNode, full)
        change = self._logic.commit_stroke(w, axis=0, idx=k,
                                           before_slice=before_2d, source=source)
        w._history.append([source, change])
        return change

    def _do_point_placement(self, w, col, row, k):
        from SegmentHumanBody import SegmentHumanBodyWidget
        # Position slice at k then fire the point.
        _setup_red_slice_at_k(self._volNode, k)
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, col, row, k)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)
        return posNode

    # ------------------------------------------------------------------

    def test_brush_then_point_first_undo_removes_only_point(self):
        """Session: brush paints slice 4 → point paints slice 5 left-half.
        First Ctrl+Z must undo only the point; the brush stroke on slice 4
        must remain intact.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()

        brush_after = np.zeros((_J, _I), dtype=np.uint8)
        brush_after[:, :_I // 2] = 1
        self._do_brush_stroke(w, self._BRUSH_K, brush_after)

        self._do_point_placement(w, col=4, row=3, k=self._POINT_K)

        # history: [brush-entry, point-entry]
        self.assertEqual(len(w._history), 2)

        SegmentHumanBodyWidget.onUndo(w)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._BRUSH_K, :, :_I // 2], 1,
            "Brush stroke on slice 4 must survive the first undo")
        np.testing.assert_array_equal(result[self._POINT_K], 0,
            "Point placement on slice 5 must be undone by the first Ctrl+Z")

    def test_brush_then_point_second_undo_removes_brush(self):
        """Continuing from the previous scenario: a second Ctrl+Z removes the
        brush stroke, leaving the mask entirely empty."""
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()

        brush_after = np.zeros((_J, _I), dtype=np.uint8)
        brush_after[:, :_I // 2] = 1
        self._do_brush_stroke(w, self._BRUSH_K, brush_after)
        self._do_point_placement(w, col=4, row=3, k=self._POINT_K)

        SegmentHumanBodyWidget.onUndo(w)   # undo point
        SegmentHumanBodyWidget.onUndo(w)   # undo brush

        result = _read_lm(self._segNode, self._segID, self._volNode)
        self.assertEqual(result.sum(), 0,
            "Mask must be entirely empty after undoing both actions")

    def test_point_then_brush_first_undo_removes_only_brush(self):
        """Session in reverse order: point first, then brush.
        First Ctrl+Z must undo only the brush; the point paint on slice 5
        must survive.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._w()

        # Place point on slice 5 first.
        self._do_point_placement(w, col=4, row=3, k=self._POINT_K)

        # Then paint a brush stroke on slice 4.
        brush_after = np.zeros((_J, _I), dtype=np.uint8)
        brush_after[:, :_I // 2] = 1
        self._do_brush_stroke(w, self._BRUSH_K, brush_after)

        # history: [point-entry, brush-entry]
        self.assertEqual(len(w._history), 2)

        SegmentHumanBodyWidget.onUndo(w)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._POINT_K, :, :_I // 2], 1,
            "Point placement on slice 5 must survive the first undo")
        np.testing.assert_array_equal(result[self._BRUSH_K], 0,
            "Brush stroke on slice 4 must be undone by the first Ctrl+Z")


# ===========================================================================
# CT-like synthetic volume helper (shared by the new workflow tests)
# ===========================================================================

def _make_ct_volume():
    """Synthetic CT-like volume with a soft-tissue background and a bright
    bone-like stripe, plus mild noise so SPX algorithms find real gradients.

    Shape is (_K, _J, _I) matching all other helpers in this file.
    Intensities are in Hounsfield-unit range: soft tissue ≈ −200 HU,
    bone stripe ≈ +400 HU.
    """
    arr = np.full((_K, _J, _I), -200, dtype=np.int16)
    # Central third → bright 'bone' region.
    arr[:, _J // 3: 2 * _J // 3, :] = 400
    # Mild noise so SLIC / Felzenszwalb find meaningful edges.
    rng = np.random.default_rng(0)
    noise = rng.integers(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int32) + noise, -32768, 32767).astype(np.int16)
    return slicer.util.addVolumeFromArray(arr)


# ===========================================================================
# Segment visibility (two-checkbox system)
# ===========================================================================

class SegmentVisibilityTest(unittest.TestCase):
    """Tests for the two-checkbox segment visibility system.

    showCurrentSegmentCheckBox / Q hotkey - controls the segment being edited.
    showSegmentsCheckBox / S hotkey       - controls all other (saved) segments.

    Each test uses a real vtkMRMLSegmentationNode with a real display node so
    that SetSegmentVisibility / GetSegmentVisibility reflect actual MRML state.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        self._volNode = _make_volume()
        self._segNode, _ = _make_seg(self._volNode)
        seg = self._segNode.GetSegmentation()
        self._segID_A = seg.AddEmptySegment('SegA')
        self._segID_B = seg.AddEmptySegment('SegB')

    def tearDown(self):
        slicer.mrmlScene.Clear()

    def _w(self, current_id=None):
        from core._state import WidgetState

        class _W:
            def _apply_saved_segments_visibility(self, exclude=None):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._apply_saved_segments_visibility(self, exclude=exclude)
            def _sync_annotation_visibility(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._sync_annotation_visibility(self)

        w = _W()
        w.ctrl = WidgetState(w)
        w._saved_segments_visible  = False
        w._current_segment_visible = True
        w._acknowledged_segment_id = None
        w.modelFamily              = None
        w.ui = MagicMock()
        w.ui.segmentationNodeSelector.currentNode.return_value = self._segNode
        w.ui.segmentSelector.currentSegmentID.return_value = (
            current_id if current_id is not None else self._segID_A
        )
        return w

    # -- onToggleCurrentSegment --

    def test_toggle_current_hides_current_segment(self):
        """visible=False hides only the current segment and updates state."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        SegmentHumanBodyWidget.onToggleCurrentSegment(w, visible=False)
        dn = self._segNode.GetDisplayNode()
        self.assertFalse(dn.GetSegmentVisibility(self._segID_A))
        self.assertFalse(w._current_segment_visible)

    def test_toggle_current_shows_current_segment(self):
        """visible=True shows a previously hidden current segment."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        w._current_segment_visible = False
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_A, False)
        SegmentHumanBodyWidget.onToggleCurrentSegment(w, visible=True)
        self.assertTrue(dn.GetSegmentVisibility(self._segID_A))
        self.assertTrue(w._current_segment_visible)

    def test_toggle_current_flips_on_each_call(self):
        """visible=None (Q hotkey path) alternates between hidden and shown."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        dn = self._segNode.GetDisplayNode()
        SegmentHumanBodyWidget.onToggleCurrentSegment(w)   # flip → hidden
        self.assertFalse(dn.GetSegmentVisibility(self._segID_A))
        SegmentHumanBodyWidget.onToggleCurrentSegment(w)   # flip → shown
        self.assertTrue(dn.GetSegmentVisibility(self._segID_A))

    def test_toggle_current_does_not_affect_other_segments(self):
        """Hiding SegA must leave SegB's display state untouched."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_B, True)
        SegmentHumanBodyWidget.onToggleCurrentSegment(w, visible=False)
        self.assertTrue(dn.GetSegmentVisibility(self._segID_B),
            "SegB visibility must not change when the current (SegA) is toggled")

    def test_toggle_current_no_segnode_is_graceful(self):
        """onToggleCurrentSegment must not raise when no segmentation is selected."""
        from core._state import WidgetState

        class _W:
            def _sync_annotation_visibility(self):
                from SegmentHumanBody import SegmentHumanBodyWidget
                SegmentHumanBodyWidget._sync_annotation_visibility(self)

        w = _W()
        w.ctrl = WidgetState(w)
        w._saved_segments_visible  = False
        w._current_segment_visible = True
        w._acknowledged_segment_id = None
        w.modelFamily              = None
        w.ui = MagicMock()
        w.ui.segmentationNodeSelector.currentNode.return_value = None
        w.ui.segmentSelector.currentSegmentID.return_value = ''
        from SegmentHumanBody import SegmentHumanBodyWidget
        try:
            SegmentHumanBodyWidget.onToggleCurrentSegment(w)
        except Exception as exc:
            self.fail(f"onToggleCurrentSegment raised {exc} with no segmentation node")

    # -- onToggleSavedSegments --

    def test_toggle_saved_hides_all_except_current(self):
        """Unchecking saved-segments must hide SegB but not SegA (current)."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        w._saved_segments_visible = True
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_A, True)
        dn.SetSegmentVisibility(self._segID_B, True)
        SegmentHumanBodyWidget.onToggleSavedSegments(w, visible=False)
        self.assertFalse(dn.GetSegmentVisibility(self._segID_B),
            "SegB must be hidden after unchecking saved-segments")
        self.assertTrue(dn.GetSegmentVisibility(self._segID_A),
            "Current segment (SegA) must not be hidden by saved-segments toggle")

    def test_toggle_saved_shows_all_except_current(self):
        """Checking saved-segments must reveal SegB; SegA state is unchanged."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_B, False)
        SegmentHumanBodyWidget.onToggleSavedSegments(w, visible=True)
        self.assertTrue(dn.GetSegmentVisibility(self._segID_B),
            "SegB must become visible after checking saved-segments")

    def test_toggle_saved_flips_on_each_call(self):
        """visible=None alternates the saved-segment state and applies it."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        w._saved_segments_visible = True
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_B, True)
        SegmentHumanBodyWidget.onToggleSavedSegments(w)    # flip → hidden
        self.assertFalse(dn.GetSegmentVisibility(self._segID_B))
        self.assertFalse(w._saved_segments_visible)
        SegmentHumanBodyWidget.onToggleSavedSegments(w)    # flip → shown
        self.assertTrue(dn.GetSegmentVisibility(self._segID_B))
        self.assertTrue(w._saved_segments_visible)

    def test_three_segments_saved_hides_both_non_current(self):
        """With three segments, toggle saved must hide B and C but not A (current)."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        seg = self._segNode.GetSegmentation()
        segID_C = seg.AddEmptySegment('SegC')
        w = self._w(self._segID_A)
        w._saved_segments_visible = True
        dn = self._segNode.GetDisplayNode()
        for sid in [self._segID_A, self._segID_B, segID_C]:
            dn.SetSegmentVisibility(sid, True)
        SegmentHumanBodyWidget.onToggleSavedSegments(w, visible=False)
        self.assertFalse(dn.GetSegmentVisibility(self._segID_B), "SegB must be hidden")
        self.assertFalse(dn.GetSegmentVisibility(segID_C),       "SegC must be hidden")
        self.assertTrue(dn.GetSegmentVisibility(self._segID_A),  "SegA must stay visible")

    # -- _apply_saved_segments_visibility --

    def test_apply_saved_excludes_current_touches_others(self):
        """_apply_saved(exclude=SegA) must set SegB to _saved_segments_visible
        while leaving SegA's state untouched."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        w._saved_segments_visible = False
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_A, True)
        dn.SetSegmentVisibility(self._segID_B, True)
        SegmentHumanBodyWidget._apply_saved_segments_visibility(w, exclude=self._segID_A)
        self.assertFalse(dn.GetSegmentVisibility(self._segID_B))
        self.assertTrue(dn.GetSegmentVisibility(self._segID_A))

    def test_apply_saved_no_exclude_affects_all_segments(self):
        """_apply_saved(exclude=None) applies to every segment including SegA."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w(self._segID_A)
        w._saved_segments_visible = False
        dn = self._segNode.GetDisplayNode()
        dn.SetSegmentVisibility(self._segID_A, True)
        dn.SetSegmentVisibility(self._segID_B, True)
        SegmentHumanBodyWidget._apply_saved_segments_visibility(w, exclude=None)
        self.assertFalse(dn.GetSegmentVisibility(self._segID_A))
        self.assertFalse(dn.GetSegmentVisibility(self._segID_B))

    def test_apply_saved_no_segnode_is_graceful(self):
        """_apply_saved must return silently when no segmentation node is present."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = self._w()
        w.ui.segmentationNodeSelector.currentNode.return_value = None
        try:
            SegmentHumanBodyWidget._apply_saved_segments_visibility(w, exclude=None)
        except Exception as exc:
            self.fail(f"_apply_saved raised {exc} with no segmentation node")


# ===========================================================================
# Window / Level preprocessing
# ===========================================================================

class WindowLevelTest(unittest.TestCase):
    """Tests for the W/L normalization applied to model input slices.

    apply_window_level (core/utils.py) is the pure-numpy implementation.
    SegmentHumanBodyLogic.set_window_level / _apply_wl_to_slice are the
    integration points exercised during model inference.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        from core._logic import SegmentHumanBodyLogic
        self._logic = SegmentHumanBodyLogic()

    def tearDown(self):
        slicer.mrmlScene.Clear()

    # -- apply_window_level utility --

    def test_no_wl_returns_input_unchanged(self):
        """apply_window_level(img, None, None) must return the original array."""
        from core.utils import apply_window_level
        arr = np.arange(12, dtype=np.int16).reshape(3, 4)
        result = apply_window_level(arr, None, None)
        np.testing.assert_array_equal(result, arr)

    def test_values_below_lo_clip_to_zero(self):
        """Intensities at or below level−window/2 must map to 0."""
        from core.utils import apply_window_level
        # window=200, level=40 → lo=−60, hi=140
        arr = np.array([[-200, -61, -60]], dtype=np.float32)
        result = apply_window_level(arr, window=200, level=40)
        self.assertEqual(result[0, 0], 0, "−200 is below lo=−60 → 0")
        self.assertEqual(result[0, 1], 0, "−61  is below lo=−60 → 0")
        self.assertEqual(result[0, 2], 0, "−60 == lo itself → 0 (inclusive clipping)")

    def test_values_above_hi_clip_to_255(self):
        """Intensities at or above level+window/2 must map to 255."""
        from core.utils import apply_window_level
        arr = np.array([[3000, 141, 140]], dtype=np.float32)
        result = apply_window_level(arr, window=200, level=40)
        self.assertEqual(result[0, 0], 255, "3000 is above hi=140 → 255")
        self.assertEqual(result[0, 1], 255, "141  is above hi=140 → 255")
        self.assertEqual(result[0, 2], 255, "140 == hi itself → 255 (inclusive clipping)")

    def test_output_range_is_0_to_255(self):
        """For any float input the output must be within [0, 255]."""
        from core.utils import apply_window_level
        rng = np.random.default_rng(42)
        arr = rng.uniform(-2000, 4000, (30, 30)).astype(np.float32)
        result = apply_window_level(arr, window=400, level=40)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)

    def test_output_dtype_is_uint8(self):
        from core.utils import apply_window_level
        arr = np.zeros((_J, _I), dtype=np.float32)
        self.assertEqual(apply_window_level(arr, window=400, level=40).dtype,
                         np.uint8)

    def test_midpoint_intensity_maps_to_127(self):
        """A pixel exactly at *level* must map to the midpoint of [0, 255]."""
        from core.utils import apply_window_level
        arr = np.full((_J, _I), 40.0, dtype=np.float32)   # all == level
        result = apply_window_level(arr, window=400, level=40)
        np.testing.assert_array_equal(result,
                                      np.full((_J, _I), 127, dtype=np.uint8),
                                      err_msg="All-at-level slice must produce all-127 output")

    def test_source_array_is_not_modified(self):
        """apply_window_level must operate on a copy; the caller's array is unchanged."""
        from core.utils import apply_window_level
        arr = np.array([[0, 100, 500, -500]], dtype=np.float32)
        original = arr.copy()
        apply_window_level(arr, window=400, level=40)
        np.testing.assert_array_equal(arr, original)

    # -- logic integration --

    def test_logic_no_wl_passes_raw_slice(self):
        """Before set_window_level is called, _apply_wl_to_slice is a no-op."""
        arr = np.arange(12, dtype=np.int16).reshape(3, 4)
        np.testing.assert_array_equal(self._logic._apply_wl_to_slice(arr), arr)

    def test_logic_set_wl_normalizes_extremes(self):
        """After set_window_level, a value far above hi maps to 255 and far
        below lo maps to 0."""
        self._logic.set_window_level(window=400, level=40)
        arr = np.array([[3000, -1000, 40]], dtype=np.float32)
        result = self._logic._apply_wl_to_slice(arr)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0], 255, "3000 → 255")
        self.assertEqual(result[0, 1], 0,   "−1000 → 0")

    def test_logic_clear_wl_reverts_to_passthrough(self):
        """set_window_level(None, None) after a prior set must revert to no-op."""
        self._logic.set_window_level(window=400, level=40)
        self._logic.set_window_level(None, None)
        arr = np.array([[3000]], dtype=np.float32)
        result = self._logic._apply_wl_to_slice(arr)
        np.testing.assert_array_equal(result, arr)

    def test_logic_wl_does_not_write_back_to_volume(self):
        """Applying W/L must never modify the source MRML volume data."""
        volNode = _make_ct_volume()
        before = slicer.util.arrayFromVolume(volNode).copy()
        self._logic.set_window_level(window=600, level=100)
        from core.utils import apply_window_level, get_slice_from_volume
        raw = get_slice_from_volume(slicer.util.arrayFromVolume(volNode), 0, 6)
        apply_window_level(raw, window=600, level=100)
        after = slicer.util.arrayFromVolume(volNode)
        np.testing.assert_array_equal(after, before,
            "MRML volume data must be unchanged after W/L preprocessing")


# ===========================================================================
# CT-like workflow — end-to-end with realistic intensity data
# ===========================================================================

@_SKIP_LAYOUT
class SampleDataWorkflowTest(unittest.TestCase):
    """End-to-end workflow tests using a synthetic CT-like volume.

    The volume has a soft-tissue background (−200 HU) and a bright stripe
    (≈+400 HU), giving SPX algorithms real intensity gradients to work with.
    Tests cover brush deltas on non-trivial data, SPX expansion on a CT slice,
    multi-segment brush + undo, and W/L integration with point placement.
    """

    _SLICE_K = 6

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_ct_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        self._ijkToRAS = _setup_red_slice_at_k(self._volNode, self._SLICE_K)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def _w(self, modelFamily=None):
        return _make_undo_test_widget(
            self._volNode, self._segNode, self._segID,
            self._pNode, self._logic, modelFamily=modelFamily)

    # -- W/L preprocessing --

    def test_wl_normalizes_ct_range_to_uint8(self):
        """W/L on a slice spanning [−220, 420] HU must produce uint8 output."""
        from core.utils import apply_window_level, get_slice_from_volume
        volumeArray = slicer.util.arrayFromVolume(self._volNode)
        raw = get_slice_from_volume(volumeArray, 0, self._SLICE_K)
        result = apply_window_level(raw, window=600, level=100)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.array_equal(raw, result),
            "W/L must change the pixel values of a CT-range slice")

    def test_wl_integration_model_receives_normalized_image(self):
        """When W/L is set, the image reaching on_expand must be uint8 normalized,
        not the raw int16 CT values.  Verified by recording the argument."""
        from core.utils import apply_window_level, get_slice_from_volume

        self._logic.set_window_level(window=600, level=100)
        volumeArray = slicer.util.arrayFromVolume(self._volNode)
        expected_img = apply_window_level(
            get_slice_from_volume(volumeArray, 0, self._SLICE_K),
            window=600, level=100,
        )

        received = []

        class _RecordingFamily:
            model = object()
            def on_expand(self, img, **kwargs):
                received.append(img.copy())
                labels = np.ones(img.shape[:2], dtype=np.int32)
                labels[:, img.shape[1] // 2:] = 2
                return labels

        w = self._w(modelFamily=_RecordingFamily())
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, self._SLICE_K)
        posNode.AddControlPoint(ras_pt)
        from SegmentHumanBody import SegmentHumanBodyWidget
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)

        self.assertEqual(len(received), 1,
            "on_expand must be called exactly once per point placement")
        np.testing.assert_array_equal(received[0], expected_img,
            "The image passed to on_expand must be the W/L-normalized uint8 slice")

    # -- Brush on CT volume --

    def test_brush_delta_on_ct_volume_matches_painted_patch(self):
        """Brush delta on a CT volume must be the tight bbox of the painted region,
        confirming the tracker works correctly with non-zero intensity data."""
        w = self._w()
        before = np.zeros((_J, _I), dtype=np.uint8)
        after  = np.zeros((_J, _I), dtype=np.uint8)
        after[3:8, 4:12] = 1    # 5-row × 8-col patch

        full = np.zeros((_K, _J, _I), dtype=np.uint8)
        full[self._SLICE_K] = after
        _write_lm(self._segNode, self._segID, self._volNode, full)

        change = self._logic.commit_stroke(w, axis=0, idx=self._SLICE_K,
                                           before_slice=before, source='brush')

        self.assertIsNotNone(change)
        self.assertEqual(change.delta.shape, (5, 8),
            "Delta must be the 5×8 tight bounding box of the painted patch")

    def test_brush_undo_on_ct_volume_restores_slice(self):
        """Painting then undoing a brush stroke on the CT volume fully restores
        the slice — not just on a zeros volume."""
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core.utils import write_slice_to_volume

        w = self._w()
        before = np.zeros((_J, _I), dtype=np.uint8)
        after  = np.zeros((_J, _I), dtype=np.uint8)
        after[1:14, 2:16] = 1

        full = np.zeros((_K, _J, _I), dtype=np.uint8)
        write_slice_to_volume(full, after, 0, self._SLICE_K)
        _write_lm(self._segNode, self._segID, self._volNode, full)

        change = self._logic.commit_stroke(w, axis=0, idx=self._SLICE_K,
                                           before_slice=before, source='brush')
        w._history.append(['brush', change])
        SegmentHumanBodyWidget.onUndo(w)

        result = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(result[self._SLICE_K], 0,
            "Brush undo on CT volume must restore the slice to empty")

    # -- SPX expansion on CT slice --

    def test_expand_on_ct_slice_paints_nonzero_region(self):
        """expandSegWithSPX on a real CT-intensity slice must paint at least one
        pixel (regression: the tracker must accept non-zero source arrays)."""
        try:
            from skimage.segmentation import slic as _check  # noqa: F401
        except ImportError:
            self.skipTest("scikit-image not available")

        from core.models.spx import SPX_SLIC2D
        from core.utils import get_slice_from_volume, apply_window_level

        volumeArray = slicer.util.arrayFromVolume(self._volNode)
        img = get_slice_from_volume(volumeArray, 0, self._SLICE_K)
        img_wl = apply_window_level(img, window=600, level=100)

        labels = SPX_SLIC2D().forward(img=img_wl, n_segments=50)

        # Seed the mask at the bright bone-stripe region.
        seed = np.zeros((_K, _J, _I), dtype=np.uint8)
        seed[self._SLICE_K, _J // 3, _I // 2] = 1
        _write_lm(self._segNode, self._segID, self._volNode, seed)

        change = self._logic.expandSegWithSPX(
            self._segNode, self._segID, self._volNode,
            labels, axis=0, sliceIndex=self._SLICE_K,
        )
        result = _read_lm(self._segNode, self._segID, self._volNode)
        self.assertGreater(result[self._SLICE_K].sum(), 0,
            "expandSegWithSPX must paint pixels on a non-trivial CT slice")
        self.assertIsNotNone(change,
            "expandSegWithSPX must return a MaskChange for a real expansion")

    def test_expand_undo_on_ct_volume_restores_state(self):
        """Expand then undo on a CT volume must restore the pre-expand mask."""
        try:
            from skimage.segmentation import slic as _check  # noqa: F401
        except ImportError:
            self.skipTest("scikit-image not available")

        from core.models.spx import SPX_SLIC2D
        from core.utils import get_slice_from_volume, apply_window_level

        volumeArray = slicer.util.arrayFromVolume(self._volNode)
        img = get_slice_from_volume(volumeArray, 0, self._SLICE_K)
        labels = SPX_SLIC2D().forward(img=apply_window_level(img, 600, 100),
                                      n_segments=50)

        pre = np.zeros((_K, _J, _I), dtype=np.uint8)
        pre[self._SLICE_K, _J // 3, _I // 2] = 1
        _write_lm(self._segNode, self._segID, self._volNode, pre)

        change = self._logic.expandSegWithSPX(
            self._segNode, self._segID, self._volNode,
            labels, axis=0, sliceIndex=self._SLICE_K,
        )
        after_expand = _read_lm(self._segNode, self._segID, self._volNode)
        self.assertGreater(after_expand[self._SLICE_K].sum(), 1,
            "Expand must paint more than the single seed pixel")

        widget = _mock_widget(self._volNode, self._segNode, self._segID)
        self._logic.reverse_change(widget, change)

        restored = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(restored, pre,
            "reverse_change must fully restore the pre-expand mask")

    # -- Multi-segment workflow --

    def test_multi_segment_independent_brush_and_undo(self):
        """Two separate segments each get a brush stroke; undoing one must not
        affect the other.

        This is a realistic annotation session: the user paints segment A,
        switches to segment B, paints it, then undoes both in reverse order.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._logic import SegmentHumanBodyLogic
        from core.utils import write_slice_to_volume

        # Add segment B alongside the existing segment A.
        segID_B = self._segNode.GetSegmentation().AddEmptySegment('SegB')

        # Segment A: brush stroke on axial slice 4.
        logic_A = SegmentHumanBodyLogic()
        logic_A.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        w_A = _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode, logic_A)

        after_A = np.zeros((_J, _I), dtype=np.uint8)
        after_A[2:9, 1:15] = 1
        full_A = np.zeros((_K, _J, _I), dtype=np.uint8)
        write_slice_to_volume(full_A, after_A, 0, 4)
        _write_lm(self._segNode, self._segID, self._volNode, full_A)
        change_A = logic_A.commit_stroke(w_A, axis=0, idx=4,
                                         before_slice=np.zeros((_J, _I), dtype=np.uint8),
                                         source='brush')
        w_A._history.append(['brush', change_A])

        # Segment B: brush stroke on axial slice 7.
        logic_B = SegmentHumanBodyLogic()
        logic_B.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        w_B = _make_undo_test_widget(
            self._volNode, self._segNode, segID_B, self._pNode, logic_B)

        after_B = np.zeros((_J, _I), dtype=np.uint8)
        after_B[5:12, 3:10] = 1
        full_B = np.zeros((_K, _J, _I), dtype=np.uint8)
        write_slice_to_volume(full_B, after_B, 0, 7)
        _write_lm(self._segNode, segID_B, self._volNode, full_B)
        change_B = logic_B.commit_stroke(w_B, axis=0, idx=7,
                                         before_slice=np.zeros((_J, _I), dtype=np.uint8),
                                         source='brush')
        w_B._history.append(['brush', change_B])

        # Verify both are painted.
        self.assertTrue(_read_lm(self._segNode, self._segID, self._volNode)[4].any(),
                        "Segment A must have pixels on slice 4")
        self.assertTrue(_read_lm(self._segNode, segID_B,       self._volNode)[7].any(),
                        "Segment B must have pixels on slice 7")

        # Undo segment B — must not touch segment A.
        SegmentHumanBodyWidget.onUndo(w_B)
        np.testing.assert_array_equal(
            _read_lm(self._segNode, segID_B, self._volNode)[7], 0,
            "Undoing segment B must clear its slice 7")
        self.assertTrue(
            _read_lm(self._segNode, self._segID, self._volNode)[4].any(),
            "Segment A must be unaffected by undoing segment B")

        # Undo segment A.
        SegmentHumanBodyWidget.onUndo(w_A)
        np.testing.assert_array_equal(
            _read_lm(self._segNode, self._segID, self._volNode)[4], 0,
            "Undoing segment A must clear its slice 4")

    def test_point_then_expand_then_undo_all(self):
        """Realistic click-expand-undo cycle on a CT slice.

        1. Place a positive point → paints one SPX region.
        2. Run expandSegWithSPX (Expand E) → grows to all matching regions.
        3. Undo expand → back to post-point state.
        4. Undo point → mask is empty.
        """
        try:
            from skimage.segmentation import slic as _check  # noqa: F401
        except ImportError:
            self.skipTest("scikit-image not available")

        from core.models.spx import SPX_SLIC2D
        from core.utils import get_slice_from_volume, apply_window_level
        from SegmentHumanBody import SegmentHumanBodyWidget

        volumeArray  = slicer.util.arrayFromVolume(self._volNode)
        img          = get_slice_from_volume(volumeArray, 0, self._SLICE_K)
        labels       = SPX_SLIC2D().forward(img=apply_window_level(img, 600, 100),
                                            n_segments=50)
        spx_family   = _make_spx_family(labels)

        w = self._w(modelFamily=spx_family)
        self._logic.set_window_level(window=600, level=100)

        # Step 1: positive point.
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, _I // 2, _J // 3, self._SLICE_K)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)
        after_point = _read_lm(self._segNode, self._segID, self._volNode).copy()
        self.assertGreater(after_point[self._SLICE_K].sum(), 0,
            "Positive point must paint at least one pixel")

        # Step 2: expand.
        widget_for_expand = _mock_widget(self._volNode, self._segNode, self._segID)
        change_expand = self._logic.expandSegWithSPX(
            self._segNode, self._segID, self._volNode,
            labels, axis=0, sliceIndex=self._SLICE_K,
        )
        if change_expand is not None:
            w._history.append(['expand', change_expand])

        # Step 3: undo expand → should restore post-point state.
        if change_expand is not None:
            SegmentHumanBodyWidget.onUndo(w)
            after_undo_expand = _read_lm(self._segNode, self._segID, self._volNode)
            np.testing.assert_array_equal(
                after_undo_expand[self._SLICE_K], after_point[self._SLICE_K],
                "Undoing expand must restore the post-point mask state")

        # Step 4: undo point → mask empty.
        SegmentHumanBodyWidget.onUndo(w)
        final = _read_lm(self._segNode, self._segID, self._volNode)
        np.testing.assert_array_equal(final[self._SLICE_K], 0,
            "Undoing the point must restore the mask to empty")


# ===========================================================================
# Bug regressions: SPX unexpected behaviors
# ===========================================================================

class SPXBrushToggleBugTest(unittest.TestCase):
    """Bug regression: toggling the brush button OFF unconditionally
    re-enters persistent point-placement mode.

    Root cause: ``onBrushToggled(checked=False)`` always calls
    ``PointHandler().attach(self)`` regardless of what mode was active
    before the brush was turned on.  A user who switches to brush (which
    detaches point mode) and then toggles brush off is silently dropped
    back into point-placement mode without any explicit action.

    The tests below FAIL with the current code to document the bug.
    They will PASS once the fix prevents the unconditional re-attachment.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()
        from core._logic import SegmentHumanBodyLogic
        self._logic  = SegmentHumanBodyLogic()
        self._pNode  = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def test_brush_toggle_off_does_not_reenter_point_mode(self):
        """onBrushToggled(False) must not unconditionally re-attach PointHandler.

        FAILS: current code always calls PointHandler().attach(self) when
        the brush button is unchecked, switching Slicer into persistent
        point-placement mode even when the user never intended it.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget
        from core._input import BrushHandler, PointHandler

        w = _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode,
            self._logic, modelFamily=None,
        )
        # Simulate brush being the active tool (bypass attach() to avoid
        # the Segment Editor effect machinery which needs a full GUI).
        brush = BrushHandler()
        brush._mouse_filter = None
        w._active_handler = brush

        SegmentHumanBodyWidget.onBrushToggled(w, False)

        self.assertNotIsInstance(
            w._active_handler, PointHandler,
            "Toggling brush off must not unconditionally re-enter point "
            "placement mode — expected a neutral (no-tool) state",
        )


@_SKIP_LAYOUT
class SPXSpuriousSegmentChangeBugTest(unittest.TestCase):
    """Bug regression: a deferred onSegmentChanged() after point confirmation
    calls clearPrompts(), wiping the just-confirmed point from _history.

    Root cause: ``_ensure_seg_and_segment()`` (called from ``commit_point``
    when no segment exists) creates a segment with Qt signals blocked.
    In real Slicer, ``qMRMLSegmentSelectorWidget`` schedules a deferred
    ``currentSegmentChanged`` via an internal QTimer.  By the time that
    timer fires, ``blockSignals`` is False and ``ctrl.creating_segment``
    is False, so ``onSegmentChanged`` has no guard and calls
    ``clearPrompts()``, which:
      • wipes ``_history`` (Ctrl+Z stops working for confirmed points), and
      • recreates the prompt nodes (observers are now on orphaned old nodes).

    The tests simulate the deferred signal by calling ``onSegmentChanged``
    directly after ``_onPointConfirmed`` returns.  They FAIL with the
    current code and will PASS after a guard (e.g. ``is_paused`` check or
    segment-ID equality check) is added to ``onSegmentChanged``.
    """

    _TARGET_K = 5

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        from core._logic import SegmentHumanBodyLogic
        self._logic   = SegmentHumanBodyLogic()
        self._pNode   = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.ensurePromptNodesExist(self._pNode)
        self._volNode = _make_volume()
        self._segNode, self._segID = _make_seg(self._volNode, with_segment=True)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, self._segNode)
        self._ijkToRAS = _setup_red_slice_at_k(self._volNode, self._TARGET_K)

        synthetic = np.ones((_J, _I), dtype=np.int32)
        synthetic[:, _I // 2:] = 2
        self._spx_fam = _make_spx_family(synthetic)

    def tearDown(self):
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        slicer.mrmlScene.Clear()

    def _place_pos_point(self):
        from SegmentHumanBody import SegmentHumanBodyWidget
        w = _make_undo_test_widget(
            self._volNode, self._segNode, self._segID, self._pNode,
            self._logic, modelFamily=self._spx_fam,
        )
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, self._TARGET_K)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)
        return w

    def test_spurious_segment_change_does_not_clear_point_history(self):
        """A deferred onSegmentChanged(current_segID) after point confirmation
        must NOT discard the confirmed point from _history.

        FAILS: onSegmentChanged has no guard against calls with the
        already-active segment ID, so clearPrompts() always fires and
        empties _history, breaking Ctrl+Z for all confirmed points.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._place_pos_point()
        self.assertEqual(len(w._history), 1,
            "Point placement must add exactly one history entry before the test")

        # Simulate the deferred currentSegmentChanged fired by
        # qMRMLSegmentSelectorWidget after _ensure_seg_and_segment unblocks signals.
        SegmentHumanBodyWidget.onSegmentChanged(w, self._segID)

        self.assertEqual(len(w._history), 1,
            "A spurious onSegmentChanged with the already-active segment ID "
            "must not clear point history — clearPrompts() must not fire")

    def test_spurious_segment_change_does_not_replace_prompt_nodes(self):
        """clearPrompts() replaces the prompt nodes; observers on the old
        nodes are silently orphaned, so subsequent placements are invisible
        to _onPointConfirmed.

        FAILS: the deferred onSegmentChanged triggers clearPrompts() which
        calls recreatePromptNodes(), replacing posNode with a brand-new node
        that has no VTK observers attached to it.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        w = self._place_pos_point()
        posNode_id_before = self._logic.getPromptNodes(self._pNode)[0].GetID()

        # Simulate deferred event.
        SegmentHumanBodyWidget.onSegmentChanged(w, self._segID)

        posNode_id_after = self._logic.getPromptNodes(self._pNode)[0].GetID()
        self.assertEqual(posNode_id_after, posNode_id_before,
            "Spurious onSegmentChanged must not replace the positive prompt node; "
            "doing so orphans all existing _onPointConfirmed VTK observers")

    def test_first_point_on_empty_segmentation_deferred_signal_does_not_clear_history(self):
        """When no segment exists, _ensure_seg_and_segment creates one inside
        commit_point.  After blockSignals(False) qMRMLSegmentSelectorWidget
        queues a deferred currentSegmentChanged.  The new segment ID has never
        been acknowledged, so without a fix the deferred signal processes
        normally and clearPrompts() wipes _history.

        _ensure_seg_and_segment must pre-set widget._acknowledged_segment_id
        so the deferred signal is treated as a duplicate.
        """
        from SegmentHumanBody import SegmentHumanBodyWidget

        # Segmentation with NO segments.
        seg_node_empty, _ = _make_seg(self._volNode, with_segment=False)
        self._logic.setVolumeAndSegmentation(self._pNode, self._volNode, seg_node_empty)

        w = _make_undo_test_widget(
            self._volNode, seg_node_empty, '', self._pNode,
            self._logic, modelFamily=self._spx_fam,
        )
        # Override side_effect to also update the mock's return_value so that
        # commit_point can read the new segment ID after _ensure_seg_and_segment.
        def _on_set_seg_dynamic(sid):
            w.ui.segmentSelector.currentSegmentID.return_value = sid
            SegmentHumanBodyWidget.onSegmentChanged(w, sid)
        w.ui.segmentSelector.setCurrentSegmentID.side_effect = _on_set_seg_dynamic
        w.ui.segmentSelector.currentNode.return_value = seg_node_empty
        w.ui.segmentationNodeSelector.currentNode.return_value = seg_node_empty

        # Place a positive point — triggers _ensure_seg_and_segment internally.
        posNode, _ = self._logic.getPromptNodes(self._pNode)
        ras_pt = _ijk_to_ras_pt(self._ijkToRAS, 4, 3, self._TARGET_K)
        posNode.AddControlPoint(ras_pt)
        SegmentHumanBodyWidget._onPointConfirmed(w, posNode)

        self.assertEqual(len(w._history), 1,
            "Placing a point when no segment exists must add one history entry")

        # The segment ID that _ensure_seg_and_segment created.
        new_seg_id = w.ui.segmentSelector.currentSegmentID()
        self.assertTrue(new_seg_id,
            "_ensure_seg_and_segment must have set a non-empty segment ID")

        # Simulate the deferred QTimer signal from blockSignals(False).
        SegmentHumanBodyWidget.onSegmentChanged(w, new_seg_id)

        self.assertEqual(len(w._history), 1,
            "Deferred currentSegmentChanged after auto-segment creation must not "
            "clear history — _ensure_seg_and_segment must pre-acknowledge the ID")
