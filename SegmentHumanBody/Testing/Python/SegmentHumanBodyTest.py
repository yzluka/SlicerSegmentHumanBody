"""Slicer-native integration tests for SegmentHumanBodyLogic.

Run from inside 3D Slicer via:
  Developer Tools -> Run Unittests  (select SegmentHumanBodyTest)

or trigger the "Reload and Test" button in the module panel, which calls
SegmentHumanBodyTest.runTest() in SegmentHumanBody.py and delegates here.

These tests require a live Slicer process — they exercise MRML scene
operations (arrayFromSegmentBinaryLabelmap, updateSegmentBinaryLabelmapFromArray,
segmentation node lifecycle, coordinate conversion, undo, and the interactive
base mask) that the pure-Python pytest suite in tests/ cannot reach.
"""

import unittest

import numpy as np
import slicer


class SegmentHumanBodyLogicTest(unittest.TestCase):
    """Integration tests for SegmentHumanBodyLogic."""

    # Volume shape for all tests: (axial slices K, rows J, cols I) — numpy order.
    _K, _J, _I = 12, 15, 18

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setUp(self):
        slicer.mrmlScene.Clear()
        from SegmentHumanBody import SegmentHumanBodyLogic
        self._logic = SegmentHumanBodyLogic()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_volume(self):
        arr = np.zeros((self._K, self._J, self._I), dtype=np.int16)
        return slicer.util.addVolumeFromArray(arr)

    def _make_seg(self, volumeNode, with_segment=False):
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        sid = seg.GetSegmentation().AddEmptySegment('Seg1') if with_segment else None
        return seg, sid

    def _make_widget(self, volumeNode=None, segNode=None, segmentID=None, paramNode=None):
        """Return a MagicMock wired to return the given Slicer nodes."""
        from unittest.mock import MagicMock
        w = MagicMock()
        w.ui.sourceVolumeSelector.currentNode.return_value = volumeNode
        w.ui.segmentSelector.currentNode.return_value = segNode
        w.ui.segmentSelector.currentSegmentID.return_value = segmentID
        w._parameterNode = paramNode
        return w

    def _write_labelmap(self, segNode, segmentID, volumeNode, arr):
        slicer.util.updateSegmentBinaryLabelmapFromArray(arr, segNode, segmentID, volumeNode)

    def _read_labelmap(self, segNode, segmentID, volumeNode):
        return slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)

    # ------------------------------------------------------------------
    # applyResult
    # ------------------------------------------------------------------

    def test_apply_result_auto_creates_segment(self):
        """applyResult auto-creates a segment when the segmentation is empty."""
        volumeNode = self._make_volume()
        segNode, _ = self._make_seg(volumeNode, with_segment=False)
        paramNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        self._logic.setVolumeAndSegmentation(paramNode, volumeNode, segNode)

        widget = self._make_widget(volumeNode, segNode, segmentID=None, paramNode=paramNode)
        mask2d = np.ones((self._J, self._I), dtype=np.uint8)
        self._logic.applyResult(widget, mask2d, axis=0, sliceIndex=5)

        self.assertEqual(segNode.GetSegmentation().GetNumberOfSegments(), 1,
                         "applyResult must create exactly one segment when none exists")

    def test_apply_result_writes_correct_axial_slice(self):
        """applyResult writes mask2d only to the requested axial slice index."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)
        widget = self._make_widget(volumeNode, segNode, segmentID)

        mask2d = np.ones((self._J, self._I), dtype=np.uint8)
        target = 4
        self._logic.applyResult(widget, mask2d, axis=0, sliceIndex=target)

        result = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[target], 1,
                                      err_msg="target slice should be all-ones")
        np.testing.assert_array_equal(result[:target], 0,
                                      err_msg="slices before target must be zero")
        np.testing.assert_array_equal(result[target + 1:], 0,
                                      err_msg="slices after target must be zero")

    def test_apply_result_working_mask_cached_across_frames(self):
        """A second applyResult call on the same segment reuses _working_mask."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)
        widget = self._make_widget(volumeNode, segNode, segmentID)

        mask_a = np.ones((self._J, self._I), dtype=np.uint8)
        mask_b = np.zeros((self._J, self._I), dtype=np.uint8)
        mask_b[:, : self._I // 2] = 1

        self._logic.applyResult(widget, mask_a, axis=0, sliceIndex=2)
        cached_id = id(self._logic._working_mask)

        self._logic.applyResult(widget, mask_b, axis=0, sliceIndex=3)
        self.assertEqual(id(self._logic._working_mask), cached_id,
                         "_working_mask must be reused (not reallocated) between frames")

        result = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1)
        np.testing.assert_array_equal(result[3, :, : self._I // 2], 1)
        np.testing.assert_array_equal(result[3, :, self._I // 2 :], 0)

    # ------------------------------------------------------------------
    # expandSegWithSPX
    # ------------------------------------------------------------------

    def test_expand_seg_with_spx_expands_matched_labels(self):
        """expandSegWithSPX selects all pixels whose SPX label is touched by the mask."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        # Paint the left half of slice 3.
        base = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        base[3, :, : self._I // 2] = 1
        self._write_labelmap(segNode, segmentID, volumeNode, base)

        # SPX: left half = label 1, right half = label 2.
        labels = np.ones((self._J, self._I), dtype=np.int32)
        labels[:, self._I // 2 :] = 2

        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        result = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3, :, : self._I // 2], 1,
                                      err_msg="label-1 region should be kept")
        np.testing.assert_array_equal(result[3, :, self._I // 2 :], 0,
                                      err_msg="label-2 region should not be added (not in mask)")

    def test_expand_seg_with_spx_neg_points_subtract(self):
        """neg_points passed to expandSegWithSPX remove their SPX labels from the result."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        # Paint entire slice 3.
        base = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        base[3] = 1
        self._write_labelmap(segNode, segmentID, volumeNode, base)

        labels = np.ones((self._J, self._I), dtype=np.int32)
        labels[:, self._I // 2 :] = 2

        # Neg point falls in label-2 territory → right half excluded.
        neg_pts = [[self._I // 2 + 1, 0]]
        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3,
                                     neg_points=neg_pts)

        result = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[3, :, : self._I // 2], 1,
                                      err_msg="label-1 region should be kept")
        np.testing.assert_array_equal(result[3, :, self._I // 2 :], 0,
                                      err_msg="label-2 region should be erased by neg point")

    def test_expand_seg_with_spx_preserves_other_slices(self):
        """expandSegWithSPX only modifies the target slice; other slices are unchanged."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        base = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        base[2] = 1                        # slice 2 painted
        base[3, :, : self._I // 2] = 1    # slice 3 left half painted
        self._write_labelmap(segNode, segmentID, volumeNode, base)

        labels = np.ones((self._J, self._I), dtype=np.int32)
        labels[:, self._I // 2 :] = 2

        # Operate only on slice 3.
        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=3)

        result = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(result[2], 1,
                                      err_msg="slice 2 must be untouched")
        # Slice 3: label 1 (left half) covered → kept; label 2 not in mask → 0.
        np.testing.assert_array_equal(result[3, :, : self._I // 2], 1)
        np.testing.assert_array_equal(result[3, :, self._I // 2 :], 0)

    # ------------------------------------------------------------------
    # undo
    # ------------------------------------------------------------------

    def test_undo_restores_pre_expand_state(self):
        """undo() reverts the slice written by expandSegWithSPX to its prior state."""
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)
        paramNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')

        # Pre-paint left half of slice 2.
        base = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        base[2, :, : self._I // 2] = 1
        self._write_labelmap(segNode, segmentID, volumeNode, base)

        # SPX: one label covers the whole slice → expand paints the whole slice.
        labels = np.ones((self._J, self._I), dtype=np.int32)
        self._logic.expandSegWithSPX(segNode, segmentID, volumeNode,
                                     labels, axis=0, sliceIndex=2)

        after_expand = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(after_expand[2], 1,
                                      err_msg="whole slice should be painted after expand")

        # Flush the working-mask cache so undo reads fresh data from Slicer.
        self._logic.reset_render_state()
        widget = self._make_widget(volumeNode, segNode, segmentID, paramNode)
        self._logic.undo(widget)

        restored = self._read_labelmap(segNode, segmentID, volumeNode)
        np.testing.assert_array_equal(restored[2, :, : self._I // 2], 1,
                                      err_msg="left half should be restored")
        np.testing.assert_array_equal(restored[2, :, self._I // 2 :], 0,
                                      err_msg="right half should be empty after undo")

    # ------------------------------------------------------------------
    # interactive base mask
    # ------------------------------------------------------------------

    def test_on_enter_interactive_snapshots_labelmap(self):
        """on_enter_interactive stores the segment's current labelmap as _interactive_base_mask."""
        from unittest.mock import MagicMock
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        existing = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        existing[5] = 1   # slice 5 fully painted
        self._write_labelmap(segNode, segmentID, volumeNode, existing)

        widget = self._make_widget(volumeNode, segNode, segmentID)
        # Provide a no-op renderer so on_enter_interactive doesn't start a QTimer.
        widget.renderer = MagicMock()

        self._logic.on_enter_interactive(widget)

        self.assertIsNotNone(self._logic._interactive_base_mask)
        np.testing.assert_array_equal(
            self._logic._interactive_base_mask, existing,
            err_msg="_interactive_base_mask must match the segment labelmap at entry")

    def test_reset_render_state_clears_base_mask(self):
        """reset_render_state() clears _interactive_base_mask set by on_enter_interactive."""
        from unittest.mock import MagicMock
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        widget = self._make_widget(volumeNode, segNode, segmentID)
        widget.renderer = MagicMock()
        self._logic.on_enter_interactive(widget)
        self.assertIsNotNone(self._logic._interactive_base_mask)

        self._logic.reset_render_state()
        self.assertIsNone(self._logic._interactive_base_mask)

    def test_ensure_interactive_base_mask_lazy_reload_after_segment_switch(self):
        """_ensure_interactive_base_mask reloads _interactive_base_mask when None.

        Simulates: user switches segments while renderer is running.
        onSegmentChanged → reset_render_state() sets _interactive_base_mask = None.
        The next render must reload it from the current segment so neg prompts
        continue to erase existing painted data.
        """
        from unittest.mock import MagicMock
        volumeNode = self._make_volume()
        segNode, segmentID = self._make_seg(volumeNode, with_segment=True)

        existing = np.zeros((self._K, self._J, self._I), dtype=np.uint8)
        existing[2] = 1
        self._write_labelmap(segNode, segmentID, volumeNode, existing)

        widget = self._make_widget(volumeNode, segNode, segmentID)
        widget.renderer = MagicMock()
        self._logic.on_enter_interactive(widget)
        self.assertIsNotNone(self._logic._interactive_base_mask)

        # Simulate segment switch clearing the snapshot.
        self._logic.reset_render_state()
        self.assertIsNone(self._logic._interactive_base_mask)

        # Lazy reload must restore the snapshot from the current segment.
        self._logic._ensure_interactive_base_mask(widget, volumeNode)

        self.assertIsNotNone(self._logic._interactive_base_mask,
                             "_interactive_base_mask must be reloaded after segment switch")
        np.testing.assert_array_equal(
            self._logic._interactive_base_mask, existing,
            err_msg="Reloaded base mask must match the segment labelmap")

    # ------------------------------------------------------------------
    # coordinate conversion
    # ------------------------------------------------------------------

    def test_ras_to_ijk_maps_volume_centre_to_middle_voxel(self):
        """_ras_to_ijk converts the RAS centre of the volume to the middle voxel index."""
        volumeNode = self._make_volume()
        # addVolumeFromArray uses 1 mm isotropic spacing with the corner at the
        # origin.  Retrieve the actual centre in RAS from the volume bounds.
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
        # pt is a 2-D pixel coordinate [col, row] for axis=0 (axial).
        self.assertAlmostEqual(pt[0], self._I / 2, delta=1,
                               msg="column index should be near volume centre")
        self.assertAlmostEqual(pt[1], self._J / 2, delta=1,
                               msg="row index should be near volume centre")

    # ------------------------------------------------------------------
    # getAxisAndSlice
    # ------------------------------------------------------------------

    def test_get_axis_and_slice_uses_volume_geometry(self):
        """getAxisAndSlice derives the voxel index from the volume's IJK-to-RAS matrix.

        The test does NOT register the volume as the background of the slice
        view, so the old GetSliceIndexFromOffset path would return a wrong or
        degenerate value.  The new RASToIJK path must still produce the correct
        voxel index.
        """
        import vtk

        # Make sure the standard four-up layout (Red/Green/Yellow + 3D) is active
        # so that named slice views exist in the layout manager.
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

        volumeNode = self._make_volume()   # shape (K=12, J=15, I=18), 1 mm isotropic
        target_k = 7

        # Find the RAS z-coordinate that corresponds to voxel K=target_k.
        # Using the volume's own IJK-to-RAS matrix makes this independent of
        # any assumption about origin or spacing.
        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        ras = ijkToRAS.MultiplyPoint([0.0, 0.0, float(target_k), 1.0])
        ras_z = ras[2]   # for the axial (Red) view the offset equals the RAS z

        # Move the Red slice to that offset without touching its background volume.
        lm = slicer.app.layoutManager()
        lm.sliceWidget("Red").mrmlSliceNode().SetSliceOffset(ras_z)

        from unittest.mock import MagicMock
        widget = MagicMock()
        widget.currentViewName = "Red"

        axis, sliceIndex = self._logic.getAxisAndSlice(widget, volumeNode)

        self.assertEqual(axis, 0,
                         "Red view must be axis 0 (axial)")
        self.assertEqual(sliceIndex, target_k,
                         "Slice index must equal the voxel index derived from the "
                         "volume's IJK-to-RAS matrix, not from GetSliceIndexFromOffset")


if __name__ == '__main__':
    unittest.main()
