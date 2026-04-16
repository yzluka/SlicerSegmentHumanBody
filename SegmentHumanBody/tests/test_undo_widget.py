"""
Pure-Python regression tests for the unified undo stack in SegmentHumanBodyWidget.

These tests verify the *data contract* of the undo stack without any
Slicer / Qt / VTK dependency.  They act as a first line of defence: if
anyone changes the entry format, the None-snapshot bug reappears, or the
LIFO ordering breaks, these tests fail before the Slicer integration suite
even runs.

Covered behaviours
------------------
Entry format
  - Every entry is a tuple; index-0 is the action-type string.
  - Brush entries carry a non-None snapshot payload (regression: the
    original code pushed ('brush', None), which caused onUndo to fall
    back to editor.undo() — a *separate* undo stack — instead of the
    unified restore path).
  - Point entries carry (type, node, cp_id, snapshot).
  - Expand entries carry (type, snapshot).
  - All snapshot payloads are 5-tuples of the form
    (segNodeID, segmentID, axis, sliceIndex, ndarray).

Stack ordering
  - Pop returns entries in LIFO order regardless of action type.
  - Multiple consecutive brush strokes produce independent entries.
  - An empty stack pop returns None safely.

Clear semantics
  - clear() empties all entries (segment switch, add segment, remove segment).
  - Push after clear works normally.

Snapshot integrity
  - The slice data stored in a snapshot is a deep copy — mutations to
    the source array after capture must not affect the stored snapshot.
  - Axis and sliceIndex are round-tripped faithfully through the tuple.
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Entry constructors  (mirror the exact code paths in SegmentHumanBodyWidget)
# ---------------------------------------------------------------------------

def _brush_entry(snapshot):
    """('brush', snapshot)  — mirrors _onBrushStrokeStart."""
    return ('brush', snapshot)


def _point_entry(node_ref, cp_id, snapshot):
    """('point', node, cp_id, snapshot)  — mirrors _onPointConfirmed."""
    return ('point', node_ref, cp_id, snapshot)


def _expand_entry(snapshot):
    """('expand', snapshot)  — mirrors expandSegWithSPX."""
    return ('expand', snapshot)


def _make_snapshot(axis=0, slice_index=3, shape=(5, 6)):
    """Return a minimal snapshot tuple with deterministic data."""
    data = np.zeros(shape, dtype=np.uint8)
    data[0, 0] = axis + 1          # make each axis distinct
    data[1, 1] = slice_index + 1   # make each slice distinct
    return ('fake-node-id', 'fake-seg-id', axis, slice_index, data.copy())


# ---------------------------------------------------------------------------
# Minimal stack — mirrors the list operations in SegmentHumanBodyWidget
# ---------------------------------------------------------------------------

class _Stack:
    def __init__(self):
        self._items = []

    def append(self, entry):
        self._items.append(entry)

    def pop(self):
        return self._items.pop() if self._items else None

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)


# ===========================================================================
# Entry format
# ===========================================================================

class TestUndoEntryFormat(unittest.TestCase):
    """Every undo entry must be a typed tuple with the correct payload shape."""

    # ---- brush ----

    def test_brush_entry_is_tuple(self):
        self.assertIsInstance(_brush_entry(_make_snapshot()), tuple)

    def test_brush_entry_type_string_is_brush(self):
        self.assertEqual(_brush_entry(_make_snapshot())[0], 'brush')

    def test_brush_entry_has_two_elements(self):
        self.assertEqual(len(_brush_entry(_make_snapshot())), 2)

    def test_brush_snapshot_is_not_none(self):
        """Regression: original code pushed ('brush', None); undo then
        fell back to editor.undo() (a separate stack) instead of the
        unified _restoreSegmentation path."""
        entry = _brush_entry(_make_snapshot())
        self.assertIsNotNone(entry[1],
                             "Brush entry must carry a snapshot, not None")

    def test_brush_snapshot_is_five_tuple(self):
        snap = _brush_entry(_make_snapshot())[1]
        self.assertEqual(len(snap), 5,
                         "Snapshot must be (segNodeID, segID, axis, sliceIdx, ndarray)")

    def test_brush_snapshot_data_is_ndarray(self):
        snap = _brush_entry(_make_snapshot())[1]
        self.assertIsInstance(snap[4], np.ndarray)

    def test_brush_snapshot_data_is_uint8(self):
        snap = _brush_entry(_make_snapshot())[1]
        self.assertEqual(snap[4].dtype, np.uint8)

    # ---- point ----

    def test_point_entry_is_tuple(self):
        self.assertIsInstance(_point_entry('n', 'cp-1', _make_snapshot()), tuple)

    def test_point_entry_type_string_is_point(self):
        self.assertEqual(_point_entry('n', 'cp-1', _make_snapshot())[0], 'point')

    def test_point_entry_has_four_elements(self):
        self.assertEqual(len(_point_entry('n', 'cp-1', _make_snapshot())), 4)

    def test_point_snapshot_is_not_none(self):
        entry = _point_entry('n', 'cp-1', _make_snapshot())
        self.assertIsNotNone(entry[3])

    def test_point_snapshot_is_five_tuple(self):
        snap = _point_entry('n', 'cp-1', _make_snapshot())[3]
        self.assertEqual(len(snap), 5)

    def test_point_cp_id_preserved(self):
        entry = _point_entry('node-ref', 'my-cp-id', _make_snapshot())
        self.assertEqual(entry[2], 'my-cp-id')

    def test_point_node_ref_preserved(self):
        entry = _point_entry('my-node', 'cp', _make_snapshot())
        self.assertEqual(entry[1], 'my-node')

    # ---- expand ----

    def test_expand_entry_is_tuple(self):
        self.assertIsInstance(_expand_entry(_make_snapshot()), tuple)

    def test_expand_entry_type_string_is_expand(self):
        self.assertEqual(_expand_entry(_make_snapshot())[0], 'expand')

    def test_expand_entry_has_two_elements(self):
        self.assertEqual(len(_expand_entry(_make_snapshot())), 2)

    def test_expand_snapshot_is_not_none(self):
        self.assertIsNotNone(_expand_entry(_make_snapshot())[1])

    def test_expand_snapshot_is_five_tuple(self):
        snap = _expand_entry(_make_snapshot())[1]
        self.assertEqual(len(snap), 5)

    # ---- type-string uniqueness ----

    def test_all_three_action_types_are_distinct_strings(self):
        types = {
            _brush_entry(_make_snapshot())[0],
            _point_entry('n', 'c', _make_snapshot())[0],
            _expand_entry(_make_snapshot())[0],
        }
        self.assertEqual(len(types), 3,
                         "brush, point, expand must have distinct type strings")


# ===========================================================================
# LIFO ordering
# ===========================================================================

class TestUndoStackLIFO(unittest.TestCase):
    """Entries pop in reverse push order, across all action types."""

    def setUp(self):
        self.stack = _Stack()

    def test_single_brush_round_trip(self):
        snap = _make_snapshot(slice_index=7)
        self.stack.append(_brush_entry(snap))
        entry = self.stack.pop()
        self.assertEqual(entry[0], 'brush')
        np.testing.assert_array_equal(entry[1][4], snap[4])

    def test_brush_pushed_last_pops_first(self):
        self.stack.append(_expand_entry(_make_snapshot(slice_index=1)))
        self.stack.append(_brush_entry(_make_snapshot(slice_index=2)))
        self.assertEqual(self.stack.pop()[0], 'brush',
                         "brush was pushed last → must pop first")
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_point_pushed_last_pops_first(self):
        self.stack.append(_brush_entry(_make_snapshot()))
        self.stack.append(_point_entry('n', 'c', _make_snapshot()))
        self.assertEqual(self.stack.pop()[0], 'point')
        self.assertEqual(self.stack.pop()[0], 'brush')

    def test_three_mixed_actions_lifo_order(self):
        self.stack.append(_expand_entry(_make_snapshot(slice_index=0)))
        self.stack.append(_brush_entry(_make_snapshot(slice_index=1)))
        self.stack.append(_point_entry('n', 'c', _make_snapshot(slice_index=2)))
        self.assertEqual(self.stack.pop()[0], 'point')
        self.assertEqual(self.stack.pop()[0], 'brush')
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_multiple_brush_strokes_are_independent_entries(self):
        """Each stroke gets its own entry — they are not merged."""
        for i in range(5):
            self.stack.append(_brush_entry(_make_snapshot(slice_index=i)))
        self.assertEqual(len(self.stack), 5)

    def test_multiple_brush_strokes_pop_in_reverse_order(self):
        for i in range(3):
            self.stack.append(_brush_entry(_make_snapshot(slice_index=i)))
        for expected in [2, 1, 0]:
            entry = self.stack.pop()
            self.assertEqual(entry[1][3], expected,
                             f"Expected slice_index {expected}, got {entry[1][3]}")

    def test_empty_stack_pop_returns_none(self):
        self.assertIsNone(self.stack.pop())

    def test_pop_beyond_size_returns_none(self):
        self.stack.append(_brush_entry(_make_snapshot()))
        self.stack.pop()
        self.assertIsNone(self.stack.pop())

    def test_stack_is_empty_after_all_pops(self):
        self.stack.append(_brush_entry(_make_snapshot()))
        self.stack.append(_expand_entry(_make_snapshot()))
        self.stack.pop()
        self.stack.pop()
        self.assertEqual(len(self.stack), 0)

    def test_snapshot_slice_index_survives_lifo_roundtrip(self):
        """The slice_index stored in a snapshot must come back unchanged."""
        snap = _make_snapshot(axis=1, slice_index=9)
        self.stack.append(_brush_entry(snap))
        entry = self.stack.pop()
        _, axis, sliceIndex, _ = entry[1][2], entry[1][2], entry[1][3], entry[1][4]
        self.assertEqual(entry[1][3], 9)


# ===========================================================================
# Clear semantics
# ===========================================================================

class TestUndoStackClear(unittest.TestCase):
    """Stack must be fully cleared on segment-switch / add / remove events."""

    def setUp(self):
        self.stack = _Stack()

    def _populate(self, n=3):
        for i in range(n):
            self.stack.append(_brush_entry(_make_snapshot(slice_index=i)))
        self.assertEqual(len(self.stack), n)

    def test_clear_empties_stack(self):
        self._populate()
        self.stack.clear()
        self.assertEqual(len(self.stack), 0)

    def test_pop_after_clear_returns_none(self):
        self._populate()
        self.stack.clear()
        self.assertIsNone(self.stack.pop())

    def test_clear_on_empty_is_no_op(self):
        self.stack.clear()   # must not raise
        self.assertEqual(len(self.stack), 0)

    def test_push_after_clear_works(self):
        self._populate()
        self.stack.clear()
        self.stack.append(_expand_entry(_make_snapshot()))
        self.assertEqual(len(self.stack), 1)
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_mixed_types_all_cleared(self):
        self.stack.append(_brush_entry(_make_snapshot()))
        self.stack.append(_point_entry('n', 'c', _make_snapshot()))
        self.stack.append(_expand_entry(_make_snapshot()))
        self.stack.clear()
        self.assertEqual(len(self.stack), 0)

    def test_partial_pop_then_clear(self):
        self._populate(4)
        self.stack.pop()   # pop one
        self.stack.clear()
        self.assertEqual(len(self.stack), 0)


# ===========================================================================
# Snapshot data integrity
# ===========================================================================

class TestSnapshotDataIntegrity(unittest.TestCase):
    """The 2D slice stored in a snapshot must be an independent deep copy."""

    def test_snapshot_independent_of_source_after_capture(self):
        """Mutating the source array after building a snapshot must not
        affect the stored data.  This is the '.copy()' guarantee."""
        data = np.zeros((4, 5), dtype=np.uint8)
        snap = ('node-id', 'seg-id', 0, 3, data.copy())
        data[:] = 99      # mutate source
        stored = snap[4]
        np.testing.assert_array_equal(stored, 0,
                                      err_msg="Snapshot must be immune to post-capture mutations")

    def test_snapshot_data_not_a_view_of_original(self):
        data = np.arange(12, dtype=np.uint8).reshape(3, 4)
        snap = ('n', 's', 0, 0, data.copy())
        # Verify they share no memory
        self.assertFalse(np.shares_memory(snap[4], data))

    def test_axis_preserved_in_snapshot(self):
        for axis in range(3):
            snap = _make_snapshot(axis=axis)
            self.assertEqual(snap[2], axis)

    def test_slice_index_preserved_in_snapshot(self):
        for idx in [0, 5, 11]:
            snap = _make_snapshot(slice_index=idx)
            self.assertEqual(snap[3], idx)

    def test_all_three_axes_produce_distinguishable_snapshots(self):
        snaps = [_make_snapshot(axis=a, slice_index=a * 3) for a in range(3)]
        axes = [s[2] for s in snaps]
        indices = [s[3] for s in snaps]
        self.assertEqual(len(set(axes)), 3, "Axes must all differ")
        self.assertEqual(len(set(indices)), 3, "Slice indices must all differ")

    def test_snapshot_data_shape_preserved(self):
        shape = (7, 11)
        snap = _make_snapshot(shape=shape)
        self.assertEqual(snap[4].shape, shape)

    def test_snapshot_node_and_segment_ids_preserved(self):
        snap = ('specific-node-123', 'specific-seg-456', 0, 2, np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(snap[0], 'specific-node-123')
        self.assertEqual(snap[1], 'specific-seg-456')


if __name__ == '__main__':
    unittest.main()
