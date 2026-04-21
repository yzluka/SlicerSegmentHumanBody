# Upcoming Sprint — TODO

## TODO 1: Point ID drift when Ctrl+Z fires while point tool is active

**Problem**  
When the user presses Ctrl+Z while the point placement tool is active, the
auto-cursor (the "Positive N" preview point) already exists in the node.
`onUndo` removes the last *confirmed* control point and may recreate the node,
but the interaction node / selection node state is not always re-synchronized
after the recreation path, which can leave the widget tracking a stale control
point ID for the active cursor.

**Expected behaviour**  
After every point undo, the active placement cursor should reflect the correct
next label (e.g. "Positive 2" → "Positive 1" after undoing the only confirmed
point), and the widget's internal cp_id bookkeeping must stay consistent so that
subsequent undos pop the right entry.

**Where to look**  
- `SegmentHumanBody.py` → `onUndo` (point branch, node-recreation path)  
- `SegmentHumanBody.py` → `_onPointConfirmed` (how cp_id is captured)  
- `core/_logic.py` → `recreate_prompt_node` (interaction node re-wiring)

---

## TODO 2: Redo functionality (Ctrl+Shift+Z)

**Design sketch**  
- Maintain a `_redo_stack` list alongside `_history`.
- On every **undo**: pop from `_history`, apply the reverse, push the entry
  onto `_redo_stack`.
- On every **redo** (Ctrl+Shift+Z): pop from `_redo_stack`, re-apply the
  forward change (via `tracker.write_slice` and re-place the control point if applicable),
  push the entry back onto `_history`.
- Any new user modification (brush stroke, erase stroke, expand, point
  confirmed) **clears `_redo_stack`** so the linear history invariant holds.

**Scope**  
All four history entry types must be redoable: `brush`, `erase`, `expand`,
`point`.  Point redo needs to re-add the control point to the markup node
(and re-trigger the render) as well as re-apply the stored `MaskChange`.

**Where to look**  
- `SegmentHumanBody.py` → `onUndo`, `_add_history`, keyboard shortcut setup  
- `core/_tracker.py` → `write_slice` (already the forward write path)  
- `core/_logic.py` → `reverse_change` (inverse; redo needs the forward version)
