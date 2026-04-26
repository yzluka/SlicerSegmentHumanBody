# Project Structure

This is a **3D Slicer scripted module** for semi-automatic medical image segmentation.
It cannot be run standalone — all execution happens inside a Slicer process, which
provides the `qt`, `vtk`, and `slicer` Python bindings.

---

## Repository layout

```
SlicerSegmentHumanBody/
├── TODO.md                          ← upcoming sprint work items
├── PROJECT_STRUCTURE.md             ← this file
└── SegmentHumanBody/                ← Slicer module root (loaded by Slicer)
    ├── SegmentHumanBody.py          ← module entry point (Widget + Test runner)
    ├── CMakeLists.txt               ← tells Slicer which Python files to install
    ├── CLAUDE.md                    ← guidance for AI assistants
    ├── Resources/
    │   └── UI/SegmentHumanBody.ui   ← Qt Designer UI file
    ├── core/                        ← pure-logic package (no Slicer imports at module level)
    │   ├── _logic.py
    │   ├── _state.py
    │   ├── _input.py
    │   ├── _tracker.py
    │   ├── modelFamilies.py
    │   ├── modelRegistry.py
    │   ├── utils.py
    │   └── models/
    │       └── spx.py               ← concrete SPX algorithm classes
    ├── models/                      ← third-party / research model code (not wired into UI yet)
    │   ├── sam/                     ← SAM v1 source
    │   ├── sam2_annotation_tool/    ← SAM 2 source + training scripts
    │   ├── segment_any_muscle/      ← research fork
    │   ├── breast_model/            ← breast segmentation model
    │   ├── ct_segmentation/         ← nnU-Net CT segmentation model
    │   └── *.py                     ← standalone architecture files (resnet, vgg, vae, …)
    ├── tests/                       ← pure-Python unit tests (no Slicer required)
    │   ├── conftest.py
    │   ├── test_families.py
    │   ├── test_registry.py
    │   ├── test_spx_models.py
    │   ├── test_undo_widget.py
    │   └── test_utils.py
    └── Testing/
        └── Python/
            └── SegmentHumanBodyTest.py  ← Slicer-native integration tests
```

---

## Core layer (`core/`)

### `SegmentHumanBody.py` — module entry point

Loaded by Slicer. Contains:

- **`SegmentHumanBody`** — `ScriptedLoadableModule` metadata (title, category, contributors).
- **`SegmentHumanBodyWidget`** — the Qt widget.  Owns the UI, the undo history list
  (`_history`), the active stroke handler (`_active_handler`), and SPX boundary overlay
  state.  Delegates all logic to the classes below.
- **`SegmentHumanBodyTest`** — `ScriptedLoadableModuleTest` integration test runner.

### `core/_logic.py` — `SegmentHumanBodyLogic`

All stateful business logic that does not belong in the widget.  Key
responsibilities:

| Attribute / method | Purpose |
|---|---|
| `_tracker` | `SegmentTracker` instance; mask cache for the current segment |
| `commit_stroke()` | Reconciles a brush/erase stroke: restores before-state, calls `tracker.write_slice()` |
| `commit_point()` | Runs SPX model, finds label at click, writes union/subtract via `tracker.write_slice()` |
| `reverse_change()` | Reverses a `MaskChange` via `tracker.reverse_delta()` |
| `on_expand()` | Runs `expandSegWithSPX` and returns the `MaskChange` for undo |
| `recreate_prompt_node()` | Replaces one markup node to reset its ID counter |
| `recreatePromptNodes()` | Replaces both markup nodes (used by `clearPrompts`) |

### `core/_state.py` — `WidgetState`

Centralises all boolean semaphores to avoid race conditions:

| Flag | Meaning |
|---|---|
| `activating_brush` | True while `StrokeHandler._activate_effect` is running |
| `brush_in_progress` | True between mouse-down and mouse-up during a stroke |
| `creating_segment` | Suppresses `onSegmentChanged` during auto-segment creation inside `applyResult` |
| `is_paused` | Nestable pause; blocks `_onPlaceModeChanged`, `_onInteractionModeChanged`, render callbacks, and `_onPointRemoved` |

`pause()` / `resume()` are nestable (depth counter).  `updateGUIFromParameterNode` wraps its two `setCurrentNode` calls inside `ctrl.pause()` so the `activeMarkupsFiducialPlaceModeChanged` signal they fire is blocked and cannot spuriously activate `PointHandler`.

### `core/_input.py` — input handler hierarchy

All three concrete handlers write through the same `SegmentTracker.write_slice()` path:

```
InputHandler          ← base (attach / detach / flush lifecycle)
├── StrokeHandler     ← owns Qt mouse filter + stroke-before snapshot
│   ├── BrushHandler  ← EFFECT='Paint', SOURCE='brush'
│   └── EraseHandler  ← EFFECT='Erase', SOURCE='erase'; skips no-op strokes
└── PointHandler      ← lifecycle only; one instance per active placement session
```

- **`_SliceViewMouseFilter`** — application-level Qt event filter; fires
  `on_press` / `on_release` callbacks on left-button events.
- **`StrokeHandler`** — captures a before-snapshot on mouse-down, commits
  via `logic.commit_stroke()` on mouse-up (through a 0-ms timer so Slicer's
  Paint effect `apply()` finishes first).  Stores the result in `_history`.
  `attach()` includes a **supersession guard**: after `_activate_effect()`
  returns it checks `widget._active_handler is self`; if not (because
  `_activate_effect` triggered `onAddSegment` which restored a fresh handler),
  it bails out without installing the mouse filter or effect callback.
- **`PointHandler`** — on each `PointPositionDefinedEvent`, calls
  `logic.commit_point()` synchronously and pushes the `MaskChange` to
  `_history` immediately (no async timer).

### `core/_tracker.py` — `SegmentTracker` + `MaskChange`

Single write path for all mask mutations.

```python
MaskChange = namedtuple('MaskChange',
    ['delta', 'r_min', 'c_min', 'axis', 'slice_idx', 'source'])
```

- `delta` is an `int16` bounding-box crop of the change (positive = added,
  negative = removed).  Storing only the bbox keeps memory per-change small
  (~200 B for a 10×10 stroke vs ~500 KB for a full 512×512 slice).
- `write_slice(axis, idx, new_data, source)` — computes delta, updates `_mask`
  in-place, pushes to Slicer, returns `MaskChange`.
- `reverse_delta(change)` — applies `−delta` to the bbox sub-region and pushes
  to Slicer; O(changed pixels).
- `get_mask()` / `get_slice()` / `snapshot()` — read API with lazy load.
- `sync()` — drops cache so next access reloads from Slicer.

### `core/modelFamilies.py` — model family classes

```
BaseModelFamily      ← VARIANTS=[], VISIBLE_BUTTONS=frozenset()
├── SAMFamily        ← SAM v1/v2 variants; onRender stub only
├── SPXModelFamily   ← superpixel algorithms; on_expand + label cache
└── AutoModelFamily  ← automated (non-interactive) segmentation
```

UI button visibility is driven entirely by `VISIBLE_BUTTONS` — the widget's
`updateUIVisibility()` shows/hides each managed button based on set membership.
Adding a button to a family = add its widget name to `VISIBLE_BUTTONS`.

`SPXModelFamily.on_expand()` runs `model.forward()` with a label-map cache keyed
on `img.ctypes.data` (O(1) pointer comparison, no copy).  Both `commit_point()`
and the Expand action go through this method, so the label map is never recomputed
for the same slice twice within one session.

`FAMILY_REGISTRY` (dict of display-name → class) is the single source of truth
for the model-family dropdown.

### `core/modelRegistry.py` — `ModelRegistry`

Lazy-instantiating session cache keyed by model name.  `get_model(name)` calls
the factory the first time and returns the cached instance thereafter.

### `core/models/spx.py` — SPX algorithm classes

Each class has `PARAM_HINT`, optional `DOC_URL`, and `forward(**kwargs)` that
pops `img` and returns an integer label map.

| Class | Algorithm |
|---|---|
| `SPX_Tester2D` | Naive uniform grid |
| `SPX_SLIC2D` | scikit-image SLIC |
| `SPX_Felzenszwalb2D` | scikit-image Felzenszwalb |

### `core/utils.py` — shared helpers

| Symbol | Purpose |
|---|---|
| `VIEW_TO_AXIS` | `{"Red": 0, "Green": 1, "Yellow": 2}` |
| `get_slice_from_volume(vol, axis, idx)` | Returns a 2-D view into the 3-D array |
| `write_slice_to_volume(vol, slice, axis, idx)` | Writes a 2-D slice in-place |
| `select_spx_labels(labels, points)` | Returns the set of SPX label IDs touched by points |
| `labels_at_points(labels, points)` | Returns SPX label values at given 2-D points |
| `call_if_exists(obj, method, **kwargs)` | Calls `obj.method(**kwargs)` if it exists, else None |
| `POSITION_DEFINED` | Slicer constant for a confirmed (placed) control point status |

---

## Mutual exclusion and handler lifecycle

### One active handler at a time

`widget._active_handler` is the single source of truth for the currently active input handler.  Every `attach()` implementation calls `_detach_current_tool_if_exists(widget)` as its first step — this flushes any pending stroke and calls `.detach()` on the previous handler before the new one is set up.

Two signals switch to `PointHandler`:

| Trigger | Path |
|---|---|
| User clicks a place-widget button | `activeMarkupsFiducialPlaceModeChanged(True)` → `_onPlaceModeChanged` |
| Slicer global toolbar enters Place mode | VTK `InteractionModeChangedEvent` → `_onInteractionModeChanged` |

Both handlers return immediately when `ctrl.is_paused` is True, so programmatic `setCurrentNode` calls (e.g. inside `updateGUIFromParameterNode` or `clearPrompts`) never spuriously activate `PointHandler`.

### Segment creation — `onAddSegment`

`clearPrompts()` (called from `onSegmentChanged` when a segment is selected or created) assigns `_active_handler = PointHandler()` directly, bypassing the `detach()` lifecycle.  To prevent it from orphaning an active stroke handler, `onAddSegment` follows a four-step sequence:

```
1. cache  — remember type(self._active_handler) if StrokeHandler
2. detach — call .detach() on current handler (flush + remove listeners)
3. create — AddEmptySegment → setCurrentSegmentID fires onSegmentChanged
               → clearPrompts → _active_handler = PointHandler()
4. restore (finally) — if prior class cached, prior_class().attach(self)
               → _detach_current_tool_if_exists removes the PointHandler
               → fresh StrokeHandler fully attached
```

The restore runs in `finally` so it also fires on early return (e.g. no volume selected), keeping the handler state consistent.

### `StrokeHandler.attach()` supersession guard

`_activate_effect()` calls `onAddSegment()` when the active segmentation has 0 segments (auto-creates the first segment so Paint has somewhere to write).  Because `onAddSegment`'s restore path calls `attach()` on a **new** handler instance, `widget._active_handler` will no longer be `self` when `_activate_effect()` returns.  The guard in `StrokeHandler.attach()`:

```python
if widget._active_handler is not self:
    return   # superseded — fresh handler already fully attached
```

prevents the original (now stale) `attach()` from installing a duplicate mouse filter and effect callback.

---

## Data flow summary

All three interactive tools write through the same single path:

```
User interaction
  │
  ├─ Brush stroke ──► StrokeHandler
  │                       capture before-state (mouse-down)
  │                       Slicer Paint effect draws
  │                       logic.commit_stroke() (mouse-up, 0-ms timer)
  │                           restore before in tracker
  │                           tracker.write_slice()  ← single write path
  │                           return MaskChange → _history
  │
  ├─ Erase stroke ──► EraseHandler  (same as Brush, source='erase')
  │
  ├─ Point placed ──► PointHandler
  │                       logic.commit_point()
  │                           SPXModelFamily.on_expand() → label map (cached)
  │                           find label at click position
  │                           tracker.write_slice()  ← single write path
  │                           return MaskChange → _history
  │
  └─ Expand (E) ──► logic.on_expand()
                        SPXModelFamily.on_expand() → label map (cached)
                        select_spx_labels (current mask drives selection)
                        tracker.write_slice()  ← single write path
                        return MaskChange → _history
```

**Undo** (`Ctrl+Z`):  
Pop from `_history` → `logic.reverse_change()` → `tracker.reverse_delta()` → mask pushed to Slicer.  
For `'point'` entries: remove control point first; recreate node if now empty (resets ID counter).

**Add Segment** (button or auto-triggered by Paint on empty segmentation):  
`onAddSegment` → cache/detach active StrokeHandler → `AddEmptySegment` → `onSegmentChanged` → `clearPrompts` → restore StrokeHandler.

---

## Coordinate system

| Space | Description |
|---|---|
| RAS | Physical space; prompt points come from Slicer in this space |
| IJK | Voxel space; used by all model code |
| 2-D slice | `axis=0` Red/axial, `axis=1` Green/coronal, `axis=2` Yellow/sagittal |

SPX 2-D point convention: `[x, y]` maps to `labels[y, x]` (row = y, col = x).

---

## Test layers

| Layer | Runner | Requires Slicer |
|---|---|---|
| `tests/` pure-Python unit tests | `PythonSlicer.exe -m pytest tests/ -v` | No |
| `Testing/Python/SegmentHumanBodyTest.py` integration tests | `PythonSlicer.exe --testing --run SegmentHumanBodyTest` | Yes |

### Pure-Python unit tests (`tests/`)

| File | Covers |
|---|---|
| `test_families.py` | SPX label cache hit/miss, `on_expand` behaviour, registry lookup |
| `test_undo_widget.py` | unified undo stack entry format, LIFO ordering, clear semantics, snapshot integrity |
| `test_registry.py` | model registry caching and factory lookup |
| `test_spx_models.py` | individual SPX model algorithms |
| `test_utils.py` | slice read/write helpers, coordinate helpers |

### Slicer-native integration tests (`Testing/Python/SegmentHumanBodyTest.py`)

| Class | Covers |
|---|---|
| `SegmentHumanBodyLogicTest` | `applyResult`, `expandSegWithSPX`, `ras_to_ijk`, `getAxisAndSlice` |
| `TrackerUndoTest` | `write_slice` / `reverse_delta` round-trips, bbox efficiency, no-op writes, LIFO undo |
| `UnifiedHistoryTest` | expand returns `MaskChange`; `reverse_change` restores state; LIFO ordering |
| `MouseFilterTest` | `_SliceViewMouseFilter` return value, callback routing, exception safety |
| `AddSegmentHandlerTest` | `onAddSegment` cache/detach/create/restore; detach order; full regression (brush → add segment → place deactivates brush); `StrokeHandler.attach()` supersession guard |
| `BrushStrokeUndoTest` | `commit_stroke` records MaskChange; `onUndo` restores slice; two-stroke LIFO; `EraseHandler._should_track` (no-op vs real erase) |
| `PointPlacementUndoTest` | `_onPointConfirmed` paints correct SPX region (pos/neg); undo removes point + restores mask; two-point LIFO; off-slice point ignored |
| `ManualPointDeletionTest` | `_onPointRemoved` reverses mask and removes history entry; suppressed while paused |
| `MixedActionUndoTest` | brush + point in one session; LIFO ordering across action types; both orderings tested |

See `CLAUDE.md` for exact run commands.
