# Upcoming Sprint — TODO

## TODO 1: Dependency checking with lazy evaluation and live cached results

**Problem**
The module has no mechanism to verify that required Python dependencies
(e.g. `scikit-image`, `torch`, model weight files) are present before a
model family is used.  Missing deps surface as cryptic import errors at
inference time.

**Design**
- Each model family / model class declares its dependencies (packages,
  files) via a class-level descriptor.
- Checking is **lazy**: a dependency is only probed the first time it is
  needed, not at module load.
- The check result is **cached in process space** (module-level dict or
  class attribute) so repeated calls are O(1) after the first probe.
- The UI surfaces a human-readable message ("Missing: torch≥2.0") rather
  than a raw traceback when a dep is absent.

**Where to look**
- `core/modelFamilies.py` → `BaseModelFamily`, `confirm_model`
- `core/modelRegistry.py` → factory lookup (good place to probe before
  instantiating)
- `SegmentHumanBody.py` → `onConfirmClicked` (display the dep error)

---

## TODO 2: "Tab" key creates a new segment

**Problem**
Adding a new segment requires clicking the "Add Segment" button.  A
keyboard shortcut would speed up the annotation workflow.

**Design**
- Register a `QShortcut` for `Tab` in `setup()`, parented to `uiWidget`
  (same pattern as `Ctrl+Z`, `E`, `Q`, `V`).
- The handler simply calls `self.onAddSegment()`.
- Ensure the shortcut does not fire while a text field (e.g.
  `paramTextEdit`) has focus — Qt's default shortcut context
  (`Qt::WindowShortcut`) already handles this for most widgets; verify
  with `brushDiameterSpinBox` and `paramTextEdit`.

**Where to look**
- `SegmentHumanBody.py` → `setup()` (shortcut registration block)
- `SegmentHumanBody.py` → `onAddSegment` (the target handler)

---

## TODO 3: Annotation log model family

**Goal**
A new model family that records, for each segment, every prompt point
the user places — along with a wall-clock timestamp — and can export the
full log as a structured list.

**Export format**
```python
[
    {
        "segment_id": str,
        "coord_ras": [x, y, z],   # 3-D world (RAS) coordinates
        "timestamp": str,          # ISO-8601, e.g. "2026-04-27T14:32:05.123"
    },
    ...
]
```

**Design**
- Subclass `BaseModelFamily` in `core/modelFamilies.py`; add to
  `FAMILY_REGISTRY`.
- The family maintains an in-memory log list.  Each entry is appended in
  `_onPointConfirmed` (or a family-specific hook) with
  `datetime.now().isoformat()`.
- The markup nodes that back the log **must not be deleted** when the user
  switches segments or calls `clearPrompts` — either store them outside
  the parameter-node reference slots or mark them as persistent.
- Expose an **Export** button (add its widget name to `VISIBLE_BUTTONS`)
  that writes the log to a user-chosen JSON or CSV file via
  `qt.QFileDialog`.
- Undo (`Ctrl+Z`) of a point must also remove the corresponding log entry
  (scan by `cp_id`).

**Where to look**
- `core/modelFamilies.py` → `BaseModelFamily`, `FAMILY_REGISTRY`
- `SegmentHumanBody.py` → `_onPointConfirmed`, `onUndo` (point branch),
  `clearPrompts` (must not wipe log nodes)
- `SegmentHumanBody.ui` → add Export button widget
