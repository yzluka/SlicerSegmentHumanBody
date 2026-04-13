import unittest
import numpy as np

from core.undoStack import UndoStack


class TestUndoStackBasic(unittest.TestCase):

    def setUp(self):
        self.stack = UndoStack()

    # ------------------------------------------------------------------ #
    # push / pop                                                           #
    # ------------------------------------------------------------------ #

    def test_pop_empty_returns_none(self):
        result = self.stack.pop('node1', 'seg1')
        self.assertIsNone(result)

    def test_push_then_pop_returns_entry(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 3, arr)
        entry = self.stack.pop('node1', 'seg1')
        self.assertIsNotNone(entry)
        axis, sliceIndex, snapshot = entry
        self.assertEqual(axis, 0)
        self.assertEqual(sliceIndex, 3)
        np.testing.assert_array_equal(snapshot, arr)

    def test_pop_is_lifo(self):
        """Last pushed entry is the first popped."""
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.ones((4, 4), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, a)
        self.stack.push('node1', 'seg1', 0, 1, b)

        _, slice_b, snap_b = self.stack.pop('node1', 'seg1')
        self.assertEqual(slice_b, 1)
        np.testing.assert_array_equal(snap_b, b)

        _, slice_a, snap_a = self.stack.pop('node1', 'seg1')
        self.assertEqual(slice_a, 0)
        np.testing.assert_array_equal(snap_a, a)

    def test_push_stores_copy(self):
        """Modifying the original array after push must not affect the snapshot."""
        arr = np.zeros((3, 3), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        arr[:] = 99     # mutate original
        _, _, snapshot = self.stack.pop('node1', 'seg1')
        np.testing.assert_array_equal(snapshot, 0)

    # ------------------------------------------------------------------ #
    # depth / has_undo                                                     #
    # ------------------------------------------------------------------ #

    def test_depth_zero_when_empty(self):
        self.assertEqual(self.stack.depth('node1', 'seg1'), 0)

    def test_depth_increments_on_push(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        self.stack.push('node1', 'seg1', 0, 1, arr)
        self.assertEqual(self.stack.depth('node1', 'seg1'), 2)

    def test_depth_decrements_on_pop(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        self.stack.pop('node1', 'seg1')
        self.assertEqual(self.stack.depth('node1', 'seg1'), 0)

    def test_has_undo_false_when_empty(self):
        self.assertFalse(self.stack.has_undo('node1', 'seg1'))

    def test_has_undo_true_after_push(self):
        self.stack.push('node1', 'seg1', 0, 0, np.zeros((2, 2), dtype=np.uint8))
        self.assertTrue(self.stack.has_undo('node1', 'seg1'))

    # ------------------------------------------------------------------ #
    # limit enforcement                                                    #
    # ------------------------------------------------------------------ #

    def test_limit_drops_oldest_entry(self):
        """When the stack is full, the oldest entry is discarded."""
        small = UndoStack(limit=3)
        arr = np.zeros((2, 2), dtype=np.uint8)
        for i in range(4):
            small.push('node1', 'seg1', 0, i, arr)

        # Only 3 entries must survive.
        self.assertEqual(small.depth('node1', 'seg1'), 3)

        # The most recent three are slices 1, 2, 3 (slice 0 was evicted).
        entry = small.pop('node1', 'seg1')
        self.assertEqual(entry[1], 3)

    def test_default_limit_is_20(self):
        self.assertEqual(self.stack._limit, UndoStack.DEFAULT_LIMIT)
        self.assertEqual(UndoStack.DEFAULT_LIMIT, 20)

    # ------------------------------------------------------------------ #
    # clear                                                                #
    # ------------------------------------------------------------------ #

    def test_clear_specific_segment(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        self.stack.push('node1', 'seg2', 0, 0, arr)

        self.stack.clear('node1', 'seg1')

        self.assertFalse(self.stack.has_undo('node1', 'seg1'))
        self.assertTrue(self.stack.has_undo('node1', 'seg2'))

    def test_clear_all(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        self.stack.push('node2', 'seg1', 0, 0, arr)

        self.stack.clear()

        self.assertFalse(self.stack.has_undo('node1', 'seg1'))
        self.assertFalse(self.stack.has_undo('node2', 'seg1'))

    def test_clear_nonexistent_key_is_noop(self):
        """Clearing a key that was never pushed must not raise."""
        self.stack.clear('ghost', 'ghost')  # no error

    # ------------------------------------------------------------------ #
    # segment isolation                                                    #
    # ------------------------------------------------------------------ #

    def test_different_segments_are_independent(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.ones((2, 2), dtype=np.uint8)

        self.stack.push('node1', 'seg1', 0, 0, a)
        self.stack.push('node1', 'seg2', 0, 0, b)

        _, _, snap1 = self.stack.pop('node1', 'seg1')
        _, _, snap2 = self.stack.pop('node1', 'seg2')

        np.testing.assert_array_equal(snap1, a)
        np.testing.assert_array_equal(snap2, b)

    def test_different_nodes_are_independent(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        self.stack.push('node1', 'seg1', 0, 0, arr)
        self.stack.push('node2', 'seg1', 0, 0, arr)

        self.assertEqual(self.stack.depth('node1', 'seg1'), 1)
        self.assertEqual(self.stack.depth('node2', 'seg1'), 1)


if __name__ == '__main__':
    unittest.main()
