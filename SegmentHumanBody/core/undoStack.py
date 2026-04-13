import numpy as np


class UndoStack:
    """Per-segment undo history for 2D slice snapshots.

    Each entry stores the state of one 2D slice *before* a destructive
    operation so it can be restored.  Stacks are keyed by
    (segNodeID, segmentID) so segment switches do not mix histories.
    """

    DEFAULT_LIMIT = 20

    def __init__(self, limit: int = DEFAULT_LIMIT):
        self._limit = limit
        self._stacks: dict = {}   # (segNodeID, segmentID) -> list[(axis, sliceIndex, ndarray)]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def push(self, seg_node_id: str, segment_id: str,
             axis: int, slice_index: int, slice_2d: np.ndarray) -> None:
        """Push a snapshot of *slice_2d* onto the stack for this segment."""
        key = (seg_node_id, segment_id)
        stack = self._stacks.setdefault(key, [])
        stack.append((axis, slice_index, slice_2d.copy()))
        if len(stack) > self._limit:
            stack.pop(0)

    def pop(self, seg_node_id: str, segment_id: str):
        """Pop and return the last snapshot, or ``None`` if the stack is empty."""
        stack = self._stacks.get((seg_node_id, segment_id))
        return stack.pop() if stack else None

    def clear(self, seg_node_id: str = None, segment_id: str = None) -> None:
        """Clear the stack for a specific segment, or all stacks if no key given."""
        if seg_node_id is not None and segment_id is not None:
            self._stacks.pop((seg_node_id, segment_id), None)
        else:
            self._stacks.clear()

    def depth(self, seg_node_id: str, segment_id: str) -> int:
        """Number of available undo steps for this segment."""
        return len(self._stacks.get((seg_node_id, segment_id), []))

    def has_undo(self, seg_node_id: str, segment_id: str) -> bool:
        return self.depth(seg_node_id, segment_id) > 0
