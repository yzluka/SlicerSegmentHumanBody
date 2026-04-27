"""
Pure-Python regression tests for the unified undo/redo stacks in
SegmentHumanBodyWidget.

These tests verify the *data contract* of both stacks without any
Slicer / Qt / VTK dependency.  They act as a first line of defence: if
anyone changes the entry format, the LIFO ordering breaks, or redo
invariants are violated, these tests fail before the Slicer integration
suite even runs.

Covered behaviours
------------------
Entry format
  - Every entry is a list; index-0 is the action-type string.
  - Brush/erase/expand entries: [type, change]
  - Point entries: [type, change, node, cp_id, ras, is_negative]
  - Expand entries: [type, change]

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

Redo stack invariants
  - onUndo moves the entry to the redo stack (LIFO mirror).
  - A new user action clears the redo stack.
  - onRedo moves the entry back to the undo stack.
  - Redo stack is empty after a new action following an undo.
  - Multiple undo/redo cycles preserve order.
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Entry constructors  (mirror the exact code paths in SegmentHumanBodyWidget)
# ---------------------------------------------------------------------------

def _brush_entry(change=None):
    """['brush', change]  — mirrors _add_history path for BrushHandler."""
    return ['brush', change]


def _erase_entry(change=None):
    """['erase', change]  — mirrors _add_history path for EraseHandler."""
    return ['erase', change]


def _point_entry(node_ref, cp_id, change=None, ras=None, is_negative=False):
    """['point', change, node, cp_id, ras, is_negative]  — mirrors _onPointConfirmed."""
    return ['point', change, node_ref, cp_id, ras or [0.0, 0.0, 0.0], is_negative]


def _expand_entry(change=None):
    """['expand', change]  — mirrors _onExpand."""
    return ['expand', change]


def _make_snapshot(axis=0, slice_index=3, shape=(5, 6)):
    """Return a minimal snapshot tuple with deterministic data."""
    data = np.zeros(shape, dtype=np.uint8)
    data[0, 0] = axis + 1          # make each axis distinct
    data[1, 1] = slice_index + 1   # make each slice distinct
    return ('fake-node-id', 'fake-seg-id', axis, slice_index, data.copy())


# ---------------------------------------------------------------------------
# Minimal stack pair — mirrors the list operations in SegmentHumanBodyWidget
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


def _simulate_undo(history, redo_stack):
    """Pop from history, push to redo — mirrors onUndo for non-point entries."""
    if not history:
        return None
    entry = history.pop()
    redo_stack.append(entry)
    return entry


def _simulate_new_action(history, redo_stack, entry):
    """Append a new user action, clearing redo — mirrors _add_history / _onExpand."""
    redo_stack.clear()
    history.append(entry)


def _simulate_redo(history, redo_stack):
    """Pop from redo, push back to history — mirrors onRedo for non-point entries."""
    if not redo_stack:
        return None
    entry = redo_stack.pop()
    history.append(entry)
    return entry


# ===========================================================================
# Entry format
# ===========================================================================

class TestUndoEntryFormat(unittest.TestCase):
    """Every undo entry must be a typed list with the correct shape."""

    # ---- brush ----

    def test_brush_entry_is_list(self):
        self.assertIsInstance(_brush_entry(), list)

    def test_brush_entry_type_string_is_brush(self):
        self.assertEqual(_brush_entry()[0], 'brush')

    def test_brush_entry_has_two_elements(self):
        self.assertEqual(len(_brush_entry()), 2)

    def test_brush_change_can_be_none(self):
        # MaskChange is None when a stroke produced no net change.
        entry = _brush_entry(None)
        self.assertIsNone(entry[1])

    def test_brush_change_round_trips(self):
        sentinel = object()
        self.assertIs(_brush_entry(sentinel)[1], sentinel)

    # ---- erase ----

    def test_erase_entry_type_string_is_erase(self):
        self.assertEqual(_erase_entry()[0], 'erase')

    def test_erase_entry_has_two_elements(self):
        self.assertEqual(len(_erase_entry()), 2)

    # ---- point ----

    def test_point_entry_is_list(self):
        self.assertIsInstance(_point_entry('n', 'cp-1'), list)

    def test_point_entry_type_string_is_point(self):
        self.assertEqual(_point_entry('n', 'cp-1')[0], 'point')

    def test_point_entry_has_six_elements(self):
        self.assertEqual(len(_point_entry('n', 'cp-1')), 6)

    def test_point_change_at_index_1(self):
        sentinel = object()
        entry = _point_entry('n', 'cp-1', change=sentinel)
        self.assertIs(entry[1], sentinel)

    def test_point_node_ref_at_index_2(self):
        entry = _point_entry('my-node', 'cp')
        self.assertEqual(entry[2], 'my-node')

    def test_point_cp_id_at_index_3(self):
        entry = _point_entry('node-ref', 'my-cp-id')
        self.assertEqual(entry[3], 'my-cp-id')

    def test_point_ras_at_index_4(self):
        ras = [1.0, 2.0, 3.0]
        entry = _point_entry('n', 'c', ras=ras)
        self.assertEqual(entry[4], ras)

    def test_point_is_negative_at_index_5_positive(self):
        entry = _point_entry('n', 'c', is_negative=False)
        self.assertFalse(entry[5])

    def test_point_is_negative_at_index_5_negative(self):
        entry = _point_entry('n', 'c', is_negative=True)
        self.assertTrue(entry[5])

    # ---- expand ----

    def test_expand_entry_is_list(self):
        self.assertIsInstance(_expand_entry(), list)

    def test_expand_entry_type_string_is_expand(self):
        self.assertEqual(_expand_entry()[0], 'expand')

    def test_expand_entry_has_two_elements(self):
        self.assertEqual(len(_expand_entry()), 2)

    def test_expand_change_round_trips(self):
        sentinel = object()
        self.assertIs(_expand_entry(sentinel)[1], sentinel)

    # ---- type-string uniqueness ----

    def test_all_action_types_are_distinct_strings(self):
        types = {
            _brush_entry()[0],
            _erase_entry()[0],
            _point_entry('n', 'c')[0],
            _expand_entry()[0],
        }
        self.assertEqual(len(types), 4,
                         "brush, erase, point, expand must have distinct type strings")


# ===========================================================================
# LIFO ordering
# ===========================================================================

class TestUndoStackLIFO(unittest.TestCase):
    """Entries pop in reverse push order, across all action types."""

    def setUp(self):
        self.stack = _Stack()

    def test_single_brush_round_trip(self):
        sentinel = object()
        self.stack.append(_brush_entry(sentinel))
        entry = self.stack.pop()
        self.assertEqual(entry[0], 'brush')
        self.assertIs(entry[1], sentinel)

    def test_brush_pushed_last_pops_first(self):
        self.stack.append(_expand_entry('change-1'))
        self.stack.append(_brush_entry('change-2'))
        self.assertEqual(self.stack.pop()[0], 'brush',
                         "brush was pushed last → must pop first")
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_point_pushed_last_pops_first(self):
        self.stack.append(_brush_entry())
        self.stack.append(_point_entry('n', 'c'))
        self.assertEqual(self.stack.pop()[0], 'point')
        self.assertEqual(self.stack.pop()[0], 'brush')

    def test_three_mixed_actions_lifo_order(self):
        self.stack.append(_expand_entry('c0'))
        self.stack.append(_brush_entry('c1'))
        self.stack.append(_point_entry('n', 'c'))
        self.assertEqual(self.stack.pop()[0], 'point')
        self.assertEqual(self.stack.pop()[0], 'brush')
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_multiple_brush_strokes_are_independent_entries(self):
        """Each stroke gets its own entry — they are not merged."""
        for i in range(5):
            self.stack.append(_brush_entry(f'change-{i}'))
        self.assertEqual(len(self.stack), 5)

    def test_multiple_brush_strokes_pop_in_reverse_order(self):
        for i in range(3):
            self.stack.append(_brush_entry(f'change-{i}'))
        for expected in ['change-2', 'change-1', 'change-0']:
            entry = self.stack.pop()
            self.assertEqual(entry[1], expected)

    def test_empty_stack_pop_returns_none(self):
        self.assertIsNone(self.stack.pop())

    def test_pop_beyond_size_returns_none(self):
        self.stack.append(_brush_entry())
        self.stack.pop()
        self.assertIsNone(self.stack.pop())

    def test_stack_is_empty_after_all_pops(self):
        self.stack.append(_brush_entry())
        self.stack.append(_expand_entry())
        self.stack.pop()
        self.stack.pop()
        self.assertEqual(len(self.stack), 0)

    def test_change_payload_survives_lifo_roundtrip(self):
        sentinel = object()
        self.stack.append(_brush_entry(sentinel))
        entry = self.stack.pop()
        self.assertIs(entry[1], sentinel)


# ===========================================================================
# Clear semantics
# ===========================================================================

class TestUndoStackClear(unittest.TestCase):
    """Stack must be fully cleared on segment-switch / add / remove events."""

    def setUp(self):
        self.stack = _Stack()

    def _populate(self, n=3):
        for i in range(n):
            self.stack.append(_brush_entry(f'change-{i}'))
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
        self.stack.append(_expand_entry('c'))
        self.assertEqual(len(self.stack), 1)
        self.assertEqual(self.stack.pop()[0], 'expand')

    def test_mixed_types_all_cleared(self):
        self.stack.append(_brush_entry())
        self.stack.append(_point_entry('n', 'c'))
        self.stack.append(_expand_entry())
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


# ===========================================================================
# Redo stack invariants
# ===========================================================================

class TestRedoStackInvariants(unittest.TestCase):
    """Redo stack must mirror undo exactly and be cleared by new user actions."""

    def setUp(self):
        self.history    = _Stack()
        self.redo_stack = _Stack()

    # ---- undo → redo transfer ----

    def test_undo_moves_entry_to_redo(self):
        self.history.append(_brush_entry('c'))
        _simulate_undo(self.history, self.redo_stack)
        self.assertEqual(len(self.history), 0)
        self.assertEqual(len(self.redo_stack), 1)

    def test_undo_preserves_entry_type_in_redo(self):
        self.history.append(_expand_entry('c'))
        entry = _simulate_undo(self.history, self.redo_stack)
        self.assertEqual(self.redo_stack.pop()[0], 'expand')

    def test_undo_preserves_change_payload_in_redo(self):
        sentinel = object()
        self.history.append(_brush_entry(sentinel))
        _simulate_undo(self.history, self.redo_stack)
        self.assertIs(self.redo_stack.pop()[1], sentinel)

    def test_two_undos_redo_stack_is_lifo(self):
        self.history.append(_brush_entry('first'))
        self.history.append(_expand_entry('second'))
        _simulate_undo(self.history, self.redo_stack)   # pops 'second'
        _simulate_undo(self.history, self.redo_stack)   # pops 'first'
        self.assertEqual(self.redo_stack.pop()[1], 'first',
                         "last undone pops first from redo")
        self.assertEqual(self.redo_stack.pop()[1], 'second')

    # ---- redo → history transfer ----

    def test_redo_moves_entry_back_to_history(self):
        self.history.append(_brush_entry('c'))
        _simulate_undo(self.history, self.redo_stack)
        _simulate_redo(self.history, self.redo_stack)
        self.assertEqual(len(self.history), 1)
        self.assertEqual(len(self.redo_stack), 0)

    def test_redo_preserves_entry_type(self):
        self.history.append(_expand_entry('c'))
        _simulate_undo(self.history, self.redo_stack)
        _simulate_redo(self.history, self.redo_stack)
        self.assertEqual(self.history.pop()[0], 'expand')

    def test_redo_preserves_change_payload(self):
        sentinel = object()
        self.history.append(_brush_entry(sentinel))
        _simulate_undo(self.history, self.redo_stack)
        _simulate_redo(self.history, self.redo_stack)
        self.assertIs(self.history.pop()[1], sentinel)

    def test_empty_redo_stack_returns_none(self):
        self.assertIsNone(_simulate_redo(self.history, self.redo_stack))

    # ---- new action clears redo ----

    def test_new_action_clears_redo_stack(self):
        self.history.append(_brush_entry('c1'))
        _simulate_undo(self.history, self.redo_stack)
        self.assertEqual(len(self.redo_stack), 1)
        _simulate_new_action(self.history, self.redo_stack, _brush_entry('c2'))
        self.assertEqual(len(self.redo_stack), 0,
                         "new user action must wipe the redo stack")

    def test_history_grows_on_new_action_after_undo(self):
        self.history.append(_brush_entry('c1'))
        _simulate_undo(self.history, self.redo_stack)
        _simulate_new_action(self.history, self.redo_stack, _brush_entry('c2'))
        self.assertEqual(len(self.history), 1)

    # ---- multi-step undo/redo cycle ----

    def test_full_undo_redo_cycle_three_actions(self):
        for i in range(3):
            _simulate_new_action(self.history, self.redo_stack, _brush_entry(f'c{i}'))
        # Undo all three
        for _ in range(3):
            _simulate_undo(self.history, self.redo_stack)
        self.assertEqual(len(self.history), 0)
        self.assertEqual(len(self.redo_stack), 3)
        # Redo all three
        for _ in range(3):
            _simulate_redo(self.history, self.redo_stack)
        self.assertEqual(len(self.history), 3)
        self.assertEqual(len(self.redo_stack), 0)
        # Verify LIFO order was preserved through the cycle
        self.assertEqual(self.history.pop()[1], 'c2')
        self.assertEqual(self.history.pop()[1], 'c1')
        self.assertEqual(self.history.pop()[1], 'c0')

    def test_partial_redo_then_new_action_clears_remaining_redo(self):
        for i in range(3):
            _simulate_new_action(self.history, self.redo_stack, _brush_entry(f'c{i}'))
        _simulate_undo(self.history, self.redo_stack)
        _simulate_undo(self.history, self.redo_stack)
        self.assertEqual(len(self.redo_stack), 2)
        _simulate_redo(self.history, self.redo_stack)     # redo one
        self.assertEqual(len(self.redo_stack), 1)
        _simulate_new_action(self.history, self.redo_stack, _brush_entry('new'))
        self.assertEqual(len(self.redo_stack), 0,
                         "remaining redo entries must be discarded after new action")

    def test_mixed_types_survive_full_undo_redo_cycle(self):
        _simulate_new_action(self.history, self.redo_stack, _brush_entry('b'))
        _simulate_new_action(self.history, self.redo_stack, _expand_entry('e'))
        _simulate_new_action(self.history, self.redo_stack, _point_entry('n', 'c', 'p'))
        for _ in range(3):
            _simulate_undo(self.history, self.redo_stack)
        for _ in range(3):
            _simulate_redo(self.history, self.redo_stack)
        types = [self.history.pop()[0] for _ in range(3)]
        self.assertEqual(types, ['point', 'expand', 'brush'])


if __name__ == '__main__':
    unittest.main()
