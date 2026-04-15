import unittest
import numpy as np

from core.utils import (
    call_if_exists,
    get_slice_from_volume,
    write_slice_to_volume,
    extract_connected_component,
    apply_window_level,
    spx_boundary_mask,
    labels_at_points,
    collect_confirmed_points,
    collect_preview_points,
    POSITION_UNDEFINED,
    POSITION_PREVIEW,
    POSITION_DEFINED,
)

try:
    import skimage  # noqa: F401
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# call_if_exists
# ---------------------------------------------------------------------------

class TestCallIfExists(unittest.TestCase):

    def test_calls_existing_method_and_returns_value(self):
        class Obj:
            def greet(self):
                return 42
        self.assertEqual(call_if_exists(Obj(), 'greet'), 42)

    def test_returns_none_when_method_absent(self):
        class Obj:
            pass
        self.assertIsNone(call_if_exists(Obj(), 'missing'))

    def test_returns_none_when_obj_is_none(self):
        self.assertIsNone(call_if_exists(None, 'anything'))

    def test_passes_args_and_kwargs(self):
        class Obj:
            def add(self, a, b=0):
                return a + b
        self.assertEqual(call_if_exists(Obj(), 'add', 3, b=4), 7)


# ---------------------------------------------------------------------------
# get_slice_from_volume / write_slice_to_volume
# ---------------------------------------------------------------------------

class TestGetSliceFromVolume(unittest.TestCase):

    def setUp(self):
        self.vol = np.arange(4 * 5 * 6).reshape(4, 5, 6)

    def test_axis0_returns_correct_slice(self):
        np.testing.assert_array_equal(
            get_slice_from_volume(self.vol, axis=0, slice_index=2),
            self.vol[2, :, :])

    def test_axis1_returns_correct_slice(self):
        np.testing.assert_array_equal(
            get_slice_from_volume(self.vol, axis=1, slice_index=3),
            self.vol[:, 3, :])

    def test_axis2_returns_correct_slice(self):
        np.testing.assert_array_equal(
            get_slice_from_volume(self.vol, axis=2, slice_index=1),
            self.vol[:, :, 1])

    def test_returned_slice_is_view(self):
        """Mutations on the returned slice propagate to the original array."""
        s = get_slice_from_volume(self.vol, axis=0, slice_index=0)
        s[0, 0] = -999
        self.assertEqual(self.vol[0, 0, 0], -999)


class TestWriteSliceToVolume(unittest.TestCase):

    def setUp(self):
        self.vol = np.zeros((4, 5, 6), dtype=np.int32)

    def _make_slice(self, shape, fill=1):
        return np.full(shape, fill, dtype=np.int32)

    def test_axis0_writes_correctly(self):
        s = self._make_slice((5, 6), fill=7)
        write_slice_to_volume(self.vol, s, axis=0, slice_index=1)
        np.testing.assert_array_equal(self.vol[1, :, :], s)
        self.assertTrue(np.all(self.vol[0, :, :] == 0))

    def test_axis1_writes_correctly(self):
        s = self._make_slice((4, 6), fill=3)
        write_slice_to_volume(self.vol, s, axis=1, slice_index=2)
        np.testing.assert_array_equal(self.vol[:, 2, :], s)

    def test_axis2_writes_correctly(self):
        s = self._make_slice((4, 5), fill=5)
        write_slice_to_volume(self.vol, s, axis=2, slice_index=4)
        np.testing.assert_array_equal(self.vol[:, :, 4], s)

    def test_roundtrip_get_write(self):
        original = np.random.randint(0, 255, (4, 5, 6), dtype=np.int32)
        target = np.zeros_like(original)
        for i in range(4):
            s = get_slice_from_volume(original, axis=0, slice_index=i)
            write_slice_to_volume(target, s, axis=0, slice_index=i)
        np.testing.assert_array_equal(target, original)


# ---------------------------------------------------------------------------
# extract_connected_component
# ---------------------------------------------------------------------------

class TestExtractConnectedComponent(unittest.TestCase):

    def test_seed_inside_region_returns_component(self):
        mask = np.array([
            [True,  True,  False],
            [True,  False, False],
            [False, False, True ],
        ])
        result = extract_connected_component(mask, point_xy=(0, 0))
        self.assertTrue(result[0, 0])
        self.assertTrue(result[1, 0])
        self.assertTrue(result[0, 1])
        # Isolated True at bottom-right must NOT be included
        self.assertFalse(result[2, 2])

    def test_seed_on_false_pixel_returns_empty(self):
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        result = extract_connected_component(mask, point_xy=(1, 1))
        self.assertFalse(result.any())

    def test_all_true_mask_returns_full_component(self):
        mask = np.ones((4, 4), dtype=bool)
        result = extract_connected_component(mask, point_xy=(2, 2))
        np.testing.assert_array_equal(result, mask)

    def test_4_connected_does_not_bridge_diagonal(self):
        """With 4-connectivity, diagonal True pixels are separate regions."""
        mask = np.array([
            [True,  False],
            [False, True ],
        ])
        result = extract_connected_component(mask, point_xy=(0, 0))
        self.assertTrue(result[0, 0])
        self.assertFalse(result[1, 1],
                         "Diagonal pixel must not be included under 4-connectivity")


# ---------------------------------------------------------------------------
# apply_window_level
# ---------------------------------------------------------------------------

class TestApplyWindowLevel(unittest.TestCase):

    def test_none_window_returns_original(self):
        img = np.array([[100, 200], [50, 300]], dtype=np.int16)
        result = apply_window_level(img, window=None, level=40)
        np.testing.assert_array_equal(result, img)

    def test_none_level_returns_original(self):
        img = np.array([[100, 200]], dtype=np.int16)
        result = apply_window_level(img, window=400, level=None)
        np.testing.assert_array_equal(result, img)

    def test_both_none_returns_original(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = apply_window_level(img, window=None, level=None)
        np.testing.assert_array_equal(result, img)

    def test_output_dtype_is_uint8(self):
        img = np.array([[0, 100, 200]], dtype=np.int16)
        result = apply_window_level(img, window=400, level=100)
        self.assertEqual(result.dtype, np.uint8)

    def test_pixel_at_window_centre_maps_to_midrange(self):
        """A pixel exactly at the level should map to ~127/128."""
        level, window = 40, 400  # typical CT soft tissue
        img = np.array([[level]], dtype=np.int16)
        result = apply_window_level(img, window=window, level=level)
        self.assertAlmostEqual(int(result[0, 0]), 127, delta=1)

    def test_pixels_below_range_clamp_to_0(self):
        level, window = 40, 400   # range = [-160, 240]
        img = np.array([[-500, -200]], dtype=np.int16)   # both below -160
        result = apply_window_level(img, window=window, level=level)
        np.testing.assert_array_equal(result, [[0, 0]])

    def test_pixels_above_range_clamp_to_255(self):
        level, window = 40, 400   # range = [-160, 240]
        img = np.array([[300, 1000]], dtype=np.int16)   # both above 240
        result = apply_window_level(img, window=window, level=level)
        np.testing.assert_array_equal(result, [[255, 255]])

    def test_full_range_maps_linearly(self):
        """lo → 0, hi → 255."""
        window, level = 200, 100  # lo=0, hi=200
        img = np.array([[0, 100, 200]], dtype=np.float32)
        result = apply_window_level(img, window=window, level=level)
        self.assertEqual(int(result[0, 0]), 0)
        self.assertEqual(int(result[0, 2]), 255)

    def test_does_not_modify_source(self):
        """The original array is never modified."""
        img = np.array([[100, 200, 300]], dtype=np.int16)
        original = img.copy()
        apply_window_level(img, window=400, level=40)
        np.testing.assert_array_equal(img, original)


# ---------------------------------------------------------------------------
# spx_boundary_mask
# ---------------------------------------------------------------------------

class TestSPXBoundaryMask(unittest.TestCase):

    def test_uniform_labels_produce_no_boundary(self):
        """A single-label image has no boundaries."""
        labels = np.ones((8, 8), dtype=np.int32)
        result = spx_boundary_mask(labels)
        self.assertEqual(result.sum(), 0)

    def test_two_horizontal_regions_boundary_at_row(self):
        """Two label regions separated by a horizontal seam."""
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[:3, :] = 1
        labels[3:, :] = 2
        result = spx_boundary_mask(labels)
        # Boundary pixels must lie at the seam rows 2 and/or 3
        self.assertTrue(result[2, :].any() or result[3, :].any())
        # Interior pixels far from the seam must not be marked
        self.assertEqual(result[0, :].sum(), 0)
        self.assertEqual(result[5, :].sum(), 0)

    def test_two_vertical_regions_boundary_at_col(self):
        """Two label regions separated by a vertical seam."""
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[:, :3] = 1
        labels[:, 3:] = 2
        result = spx_boundary_mask(labels)
        self.assertTrue(result[:, 2].any() or result[:, 3].any())
        self.assertEqual(result[:, 0].sum(), 0)

    def test_output_dtype_is_uint8(self):
        labels = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int32)
        result = spx_boundary_mask(labels)
        self.assertEqual(result.dtype, np.uint8)

    def test_output_values_are_binary(self):
        labels = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [3, 3, 4, 4],
                           [3, 3, 4, 4]], dtype=np.int32)
        result = spx_boundary_mask(labels)
        unique = set(result.ravel().tolist())
        self.assertTrue(unique.issubset({0, 1}),
                        f"Expected only 0 and 1, got {unique}")

    def test_output_shape_matches_input(self):
        labels = np.arange(20).reshape(4, 5).astype(np.int32)
        result = spx_boundary_mask(labels)
        self.assertEqual(result.shape, labels.shape)

    def test_checkerboard_all_pixels_are_boundaries(self):
        """In a checkerboard every pixel borders a different label."""
        rr, cc = np.mgrid[:6, :6]
        labels = ((rr + cc) % 2).astype(np.int32) + 1   # alternating 1 and 2
        result = spx_boundary_mask(labels)
        # Every pixel should be on a boundary
        self.assertTrue(result.all(),
                        "Every pixel in a checkerboard must be a boundary pixel")


# ---------------------------------------------------------------------------
# labels_at_points
# ---------------------------------------------------------------------------

class TestLabelsAtPoints(unittest.TestCase):

    def setUp(self):
        # 4-quadrant label map: each 3×3 quadrant has a unique label.
        self.labels = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ], dtype=np.int32)

    def test_single_point_returns_correct_label(self):
        self.assertEqual(labels_at_points([[1, 1]], self.labels), {1})

    def test_multiple_points_same_region_returns_one_label(self):
        result = labels_at_points([[0, 0], [1, 2], [2, 1]], self.labels)
        self.assertEqual(result, {1})

    def test_points_across_regions_return_multiple_labels(self):
        result = labels_at_points([[1, 1], [4, 1], [1, 4], [4, 4]], self.labels)
        self.assertEqual(result, {1, 2, 3, 4})

    def test_empty_points_returns_empty_set(self):
        self.assertEqual(labels_at_points([], self.labels), set())

    def test_out_of_bounds_points_are_ignored(self):
        result = labels_at_points([[-1, 0], [0, -1], [10, 0], [0, 10]], self.labels)
        self.assertEqual(result, set())

    def test_mixed_valid_and_oob_only_valid_contribute(self):
        result = labels_at_points([[1, 1], [999, 999]], self.labels)
        self.assertEqual(result, {1})

    def test_returns_set_not_list(self):
        self.assertIsInstance(labels_at_points([[0, 0]], self.labels), set)

    def test_duplicate_points_still_return_unique_labels(self):
        result = labels_at_points([[0, 0], [0, 0], [0, 0]], self.labels)
        self.assertEqual(result, {1})


# ---------------------------------------------------------------------------
# collect_confirmed_points
# ---------------------------------------------------------------------------

POS = (1.0, 2.0, 3.0)   # arbitrary RAS position reused across tests


class TestCollectConfirmedPoints(unittest.TestCase):

    def test_defined_point_is_included(self):
        records = [(POSITION_DEFINED, "id1", POS)]
        self.assertEqual(collect_confirmed_points(records, set()), [POS])

    def test_undefined_point_always_excluded(self):
        records = [(POSITION_UNDEFINED, "id1", POS)]
        self.assertEqual(collect_confirmed_points(records, {"id1"}), [])

    def test_undefined_point_excluded_even_without_preview_ids(self):
        records = [(POSITION_UNDEFINED, "id1", POS)]
        self.assertEqual(collect_confirmed_points(records, set()), [])

    def test_preview_in_preview_ids_excluded(self):
        """Unconfirmed cursor (in preview_ids) must not appear as a confirmed point."""
        records = [(POSITION_PREVIEW, "cursor1", POS)]
        self.assertEqual(collect_confirmed_points(records, {"cursor1"}), [])

    def test_preview_not_in_preview_ids_included(self):
        """Slicer builds that leave confirmed points at PositionPreview status:
        a point whose ID was removed from preview_ids (confirmed by click) must
        still be included even though its status flag is still Preview."""
        records = [(POSITION_PREVIEW, "id1", POS)]
        self.assertEqual(collect_confirmed_points(records, set()), [POS])

    def test_mixed_records_only_confirmed_returned(self):
        p1 = (1.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        p3 = (3.0, 0.0, 0.0)
        p4 = (4.0, 0.0, 0.0)
        records = [
            (POSITION_DEFINED,   "a", p1),   # included
            (POSITION_PREVIEW,   "b", p2),   # included (not in preview_ids)
            (POSITION_PREVIEW,   "c", p3),   # excluded (cursor, in preview_ids)
            (POSITION_UNDEFINED, "d", p4),   # excluded
        ]
        result = collect_confirmed_points(records, {"c"})
        self.assertEqual(result, [p1, p2])

    def test_empty_records_returns_empty(self):
        self.assertEqual(collect_confirmed_points([], {"id1"}), [])

    def test_auto_cursor_regression(self):
        """Regression: after clearPrompts, the widget auto-creates a PositionPreview
        cursor.  _onPointAdded adds its ID to _preview_cp_ids.  That cursor must
        NOT appear as a confirmed 'Positive 1' — the user hasn't placed it yet.

        In the Slicer widget, clearPrompts() also calls
        markupsNode.SetLastUsedControlPointNumber(0) before RemoveAllControlPoints()
        so the widget's auto-cursor always gets the label 'Positive 1' regardless
        of how many points were placed and removed in earlier sessions.
        """
        auto_cursor_id = "auto-cursor-99"
        auto_pos = (5.0, 5.0, 0.0)
        # Simulate the node state after clearPrompts + widget auto-cursor
        records = [(POSITION_PREVIEW, auto_cursor_id, auto_pos)]
        preview_ids = {auto_cursor_id}   # _onPointAdded correctly recorded it
        confirmed = collect_confirmed_points(records, preview_ids)
        self.assertEqual(confirmed, [],
                         "Auto-cursor created by widget after clearPrompts must not "
                         "be treated as a confirmed prompt point")

    def test_auto_cursor_becomes_confirmed_after_click(self):
        """After the user clicks, the ID is removed from preview_ids.
        The point must then appear as confirmed even if status stays Preview."""
        auto_cursor_id = "auto-cursor-99"
        auto_pos = (5.0, 5.0, 0.0)
        records = [(POSITION_PREVIEW, auto_cursor_id, auto_pos)]
        preview_ids = set()   # ID removed from preview_ids by _onPointConfirmed
        confirmed = collect_confirmed_points(records, preview_ids)
        self.assertEqual(confirmed, [auto_pos],
                         "Point confirmed by click must appear even if status is Preview")


# ---------------------------------------------------------------------------
# collect_preview_points
# ---------------------------------------------------------------------------

class TestCollectPreviewPoints(unittest.TestCase):

    def test_cursor_in_preview_ids_returned(self):
        records = [(POSITION_PREVIEW, "cur1", POS)]
        self.assertEqual(collect_preview_points(records, {"cur1"}), [POS])

    def test_defined_point_not_returned(self):
        records = [(POSITION_DEFINED, "id1", POS)]
        self.assertEqual(collect_preview_points(records, {"id1"}), [])

    def test_undefined_point_not_returned(self):
        records = [(POSITION_UNDEFINED, "id1", POS)]
        self.assertEqual(collect_preview_points(records, {"id1"}), [])

    def test_preview_not_in_preview_ids_not_returned(self):
        """A Preview-status point whose ID is not in preview_ids is already
        confirmed — it should NOT appear as a hover cursor."""
        records = [(POSITION_PREVIEW, "id1", POS)]
        self.assertEqual(collect_preview_points(records, set()), [])

    def test_multiple_cursors_all_returned(self):
        p1, p2 = (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)
        records = [
            (POSITION_PREVIEW, "c1", p1),
            (POSITION_PREVIEW, "c2", p2),
        ]
        result = collect_preview_points(records, {"c1", "c2"})
        self.assertEqual(sorted(result), sorted([p1, p2]))

    def test_empty_records_returns_empty(self):
        self.assertEqual(collect_preview_points([], {"id1"}), [])


# ---------------------------------------------------------------------------
# Negative auto-focus prevention regression tests
#
# Background
# ----------
# When clearPrompts() is called, it:
#   1. Removes (or recreates) both the positive and negative markups nodes.
#   2. Calls setCurrentNode(negNode) then setCurrentNode(posNode) on the
#      two qSlicerSimpleMarkupsWidget instances.
#   3. If either widget was previously in PersistentPlaceMode, the
#      setCurrentNode call inherits that placement mode and fires
#      _onPointAdded for an auto-cursor on the new node.
#
# The fix ensures:
#   a) SwitchToViewTransformMode() is called BEFORE setCurrentNode so no
#      auto-cursor fires on the negative node.
#   b) If an auto-cursor does appear (positive node only, by design),
#      _onPointAdded must record its ID in _preview_cp_ids immediately.
#   c) collect_confirmed_points must exclude any point whose ID is still
#      in _preview_cp_ids (it's the placement cursor, not a user click).
#
# These tests verify the collect_confirmed_points contract for the three
# concrete failure scenarios the bug introduced.
# ---------------------------------------------------------------------------

class TestNegativeAutoFocusPrevention(unittest.TestCase):
    """Regression tests for the 'enters as Negative 1' / 'starts at Positive 2'
    bugs caused by qSlicerSimpleMarkupsWidget inheriting placement mode when
    setCurrentNode() is called on a fresh node.

    All tests are pure-Python and do not require Slicer.
    """

    # ------------------------------------------------------------------
    # Case 1 (correct behaviour): negative auto-cursor is in preview_ids
    # ------------------------------------------------------------------

    def test_negative_cursor_in_preview_ids_is_not_a_confirmed_negative_prompt(self):
        """After clearPrompts the negative widget must not have a *confirmed* point.

        Scenario: clearPrompts recreates negNode; the negative widget somehow
        fires _onPointAdded (e.g. the fix is incomplete) and creates a
        PositionPreview cursor.  _onPointAdded correctly records its ID in
        _preview_cp_ids.

        Expected: collect_confirmed_points returns [] — the cursor is still
        unconfirmed; no negative prompt fires.
        """
        neg_cursor_id  = "neg-auto-cursor"
        neg_cursor_pos = (5.0, 3.0, 0.0)

        # Negative node after clearPrompts: one preview cursor
        neg_records = [(POSITION_PREVIEW, neg_cursor_id, neg_cursor_pos)]
        preview_ids  = {neg_cursor_id}   # _onPointAdded recorded it correctly

        confirmed_neg = collect_confirmed_points(neg_records, preview_ids)

        self.assertEqual(
            confirmed_neg, [],
            "A PositionPreview cursor on the negative node that is tracked in "
            "_preview_cp_ids must NOT be treated as a confirmed negative prompt."
        )

    # ------------------------------------------------------------------
    # Case 2 (failure-mode documentation): cursor missing from preview_ids
    # ------------------------------------------------------------------

    def test_negative_cursor_absent_from_preview_ids_fires_as_false_negative(self):
        """Documents the exact failure mode the bug introduced.

        If _onPointAdded is NOT called (tracking gap) or _preview_cp_ids is
        cleared BEFORE the widget auto-cursor is created, the cursor ID is
        absent from preview_ids.  collect_confirmed_points then treats a
        PositionPreview point with an unknown ID as *confirmed* — which is the
        correct fallback for Slicer builds that leave real user clicks at
        PositionPreview.  This is what caused 'enters as Negative 1'.

        This test explicitly asserts the wrong result to document the failure
        mode.  The production fix (SwitchToViewTransformMode before
        setCurrentNode) prevents the negative auto-cursor from being created
        in the first place, making this scenario unreachable at runtime.
        """
        neg_cursor_id  = "neg-auto-cursor"
        neg_cursor_pos = (5.0, 3.0, 0.0)

        neg_records = [(POSITION_PREVIEW, neg_cursor_id, neg_cursor_pos)]
        preview_ids  = set()  # ID was NOT recorded → tracking failure

        confirmed_neg = collect_confirmed_points(neg_records, preview_ids)

        # The untracked Preview point is mis-classified as confirmed.
        # This is the root cause of the 'Negative 1' regression.
        self.assertEqual(
            confirmed_neg, [neg_cursor_pos],
            "An untracked PositionPreview point (ID absent from preview_ids) "
            "is incorrectly classified as a confirmed negative prompt. "
            "This documents the failure mode — the production fix prevents "
            "the negative cursor from ever being created."
        )

    # ------------------------------------------------------------------
    # Case 3: both nodes have cursors; only positive cursor is auto-placed
    # ------------------------------------------------------------------

    def test_after_clear_prompts_only_positive_node_has_tracked_cursor(self):
        """After clearPrompts, placement is activated for the positive node only.

        positive node: 1 auto-cursor, ID in preview_ids → not confirmed
        negative node: 0 control points                 → not confirmed

        The negative node being empty is what SwitchToViewTransformMode()
        before setCurrentNode(negNode) guarantees.
        """
        pos_cursor_id  = "pos-auto-cursor"
        pos_cursor_pos = (0.0, 0.0, 0.0)

        pos_records = [(POSITION_PREVIEW, pos_cursor_id, pos_cursor_pos)]
        neg_records = []                       # negative node has no points
        preview_ids = {pos_cursor_id}

        confirmed_pos = collect_confirmed_points(pos_records, preview_ids)
        confirmed_neg = collect_confirmed_points(neg_records, preview_ids)

        self.assertEqual(confirmed_pos, [],
                         "Positive auto-cursor in preview_ids must not be confirmed.")
        self.assertEqual(confirmed_neg, [],
                         "Empty negative node must yield no confirmed negative prompts.")

    # ------------------------------------------------------------------
    # Case 4: multiple re-entries (simulate N consecutive clearPrompts calls)
    # ------------------------------------------------------------------

    def test_multiple_reentries_leave_no_stale_confirmed_points(self):
        """Simulate three consecutive clearPrompts + interactive-mode exits.

        Each entry creates fresh node IDs (recreatePromptNodes produces brand-
        new nodes with a counter starting at 0).  The final session's cursor
        is the only tracked ID; all earlier IDs are gone with the old nodes.
        Confirmed lists must be empty at the start of each new session.
        """
        sessions = [
            ("cursor-session1", (1.0, 0.0, 0.0)),
            ("cursor-session2", (2.0, 0.0, 0.0)),
            ("cursor-session3", (3.0, 0.0, 0.0)),
        ]
        for cursor_id, cursor_pos in sessions:
            pos_records = [(POSITION_PREVIEW, cursor_id, cursor_pos)]
            neg_records = []
            preview_ids = {cursor_id}

            confirmed_pos = collect_confirmed_points(pos_records, preview_ids)
            confirmed_neg = collect_confirmed_points(neg_records, preview_ids)

            self.assertEqual(
                confirmed_pos, [],
                f"Session with cursor {cursor_id!r}: auto-cursor must not be confirmed."
            )
            self.assertEqual(
                confirmed_neg, [],
                f"Session with cursor {cursor_id!r}: negative node must have no confirmed points."
            )

    # ------------------------------------------------------------------
    # Case 5: user confirms a positive click after the auto-cursor
    # ------------------------------------------------------------------

    def test_user_click_after_autocursor_produces_exactly_one_confirmed_positive(self):
        """After the user clicks once, exactly one confirmed positive point exists.

        Sequence (single session):
          1. clearPrompts → auto-cursor "cur0" created for positive node, tracked.
          2. User moves mouse to slice and clicks → "cur0" confirmed (removed from
             preview_ids); Slicer creates next auto-cursor "cur1" also tracked.

        At this point:
          pos_records = [
              (POSITION_PREVIEW,  "cur0",  click_pos),   # confirmed click
              (POSITION_PREVIEW,  "cur1",  hover_pos),   # next cursor, still live
          ]
          preview_ids = {"cur1"}   # cur0 removed on confirmation

        collect_confirmed_points should return exactly [click_pos].
        """
        click_pos = (7.0, 3.0, 0.0)
        hover_pos = (0.0, 0.0, 0.0)   # position is irrelevant for this test

        pos_records = [
            (POSITION_PREVIEW, "cur0", click_pos),   # confirmed: NOT in preview_ids
            (POSITION_PREVIEW, "cur1", hover_pos),   # live cursor: in preview_ids
        ]
        preview_ids = {"cur1"}

        confirmed = collect_confirmed_points(pos_records, preview_ids)

        self.assertEqual(
            confirmed, [click_pos],
            "After one user click there must be exactly one confirmed positive point."
        )


if __name__ == '__main__':
    unittest.main()
