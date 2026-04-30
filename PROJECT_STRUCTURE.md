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
    │   ├── _deps.py                 ← DependencyCheck: lazy, cached package/file probing
    │   ├── modelFamilies.py
    │   ├── modelRegistry.py
    │   ├── utils.py
    │   └── models/
    │       ├── spx.py               ← concrete SPX algorithm classes
    │       └── timed_annotator.py   ← TimedAnnotatorModel: per-segment timestamped log
    ├── models/                      ← third-party / research model code (not wired into UI yet)
    │   ├── sam/                     ← SAM v1 source
    │   ├── sam2_annotation_tool/    ← SAM 2 source + training scripts
    │   ├── segment_any_muscle/      ← research fork
    │   ├── breast_model/            ← breast segmentation model
    │   ├── ct_segmentation/         ← nnU-Net CT segmentation model
    │   └── *.py                     ← standalone architecture files (resnet, vgg, vae, …)
    ├── tests/                       ← pure-Python unit tests (no Slicer required)
    │   ├── conftest.py
    │   ├── test_deps.py
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
  (`_history`), the active stroke handler (`_active_handler`), SPX boundary overlay
  state, and segment visibility state (`_current_segment_visible`,
  `_saved_segments_visible`).  Delegates all logic to the classes below.
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
| `creating_segment` | Suppresses `onSegmentChanged` during auto-segment creation inside `_ensure_seg_and_segment` |
| `is_paused` | Nestable pause; blocks `_onPlaceModeChanged`, `_onInteractionModeChanged`, render callbacks, and `_onPointRemoved` |

`pause()` / `resume()` are nestable (depth counter).  `updateGUIFromParameterNode` wraps its two markup-node `setCurrentNode` calls (`positivePrompts`, `negativePrompts`) inside `ctrl.pause()` so the `activeMarkupsFiducialPlaceModeChanged` signal they fire is blocked and cannot spuriously activate `PointHandler`.  The `segmentSelector.setCurrentNode` call is **outside** the pause block.

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
  returns it checks `widget._active_handler is self`; if not, it bails out
  without installing the mouse filter or effect callback (defensive: prevents
  a stale `attach()` if a signal inside `_activate_effect` re-enters `onAddSegment`).
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
BaseModelFamily          ← VARIANTS=[], VISIBLE_BUTTONS=frozenset()
├── SAMFamily            ← SAM v1/v2 variants; onRender stub only
├── SPXModelFamily       ← superpixel algorithms; on_expand + label cache
├── AutoModelFamily      ← automated (non-interactive) segmentation
└── TimedAnnotatorFamily ← timestamped annotation log; delegates to TimedAnnotatorModel
```

UI button visibility is driven entirely by `VISIBLE_BUTTONS` — the widget's
`updateUIVisibility()` shows/hides each managed button based on set membership.
Adding a button to a family = add its widget name to `VISIBLE_BUTTONS`.

`SPXModelFamily.on_expand()` runs `model.forward()` with a label-map cache keyed
on `img.ctypes.data` (O(1) pointer comparison, no copy).  Both `commit_point()`
and the Expand action go through this method, so the label map is never recomputed
for the same slice twice within one session.

`TimedAnnotatorFamily` has no model weights — it auto-confirms on family switch.
All logic lives in `TimedAnnotatorModel` (cached by `ModelRegistry` for the
session so switching families and back preserves the accumulated log).  It
records per-segment timestamped points, mirrors them as persistent
`vtkMRMLMarkupsFiducialNode` nodes with per-segment palette colors, and supports
JSON export/import in nested per-segment format (with legacy flat-list
compatibility).

`FAMILY_REGISTRY` (dict of display-name → class) is the single source of truth
for the model-family dropdown:

| Display name | Class |
|---|---|
| `'None'` | `BaseModelFamily` |
| `'SAM-Style'` | `SAMFamily` |
| `'SPX-Assisted Annotation'` | `SPXModelFamily` |
| `'Auto'` | `AutoModelFamily` |
| `'TimedMarker'` | `TimedAnnotatorFamily` |

### `core/modelRegistry.py` — `ModelRegistry`

Lazy-instantiating session cache keyed by model name.  `get_model(name)` calls
the factory the first time and returns the cached instance thereafter.

### `core/_deps.py` — `DependencyCheck`

Process-scoped dependency checker with O(1) repeated calls.  Results (success or
error string) are stored in a class-level `_cache` dict keyed by
`(kind, *key_parts)`.  Two probe families:

| Method | Raises on failure |
|---|---|
| `require_package(import_name, *, display_name, min_version)` | `ImportError` |
| `require_file(path, *, display_name)` | `FileNotFoundError` |
| `check_package(...)` | returns `(ok: bool, message: str)` |
| `check_file(...)` | returns `(ok: bool, message: str)` |

SPX model constructors (`SPX_SLIC2D`, `SPX_Felzenszwalb2D`) call
`DependencyCheck.require_package('skimage', display_name='scikit-image')` in
their `__init__`, surfacing a human-readable error before the import fails.

### `core/models/spx.py` — SPX algorithm classes

Each class has `PARAM_HINT`, optional `DOC_URL`, and `forward(**kwargs)` that
pops `img` and returns an integer label map.  Constructors call
`DependencyCheck.require_package('skimage')` for a clean error when
scikit-image is absent.

| Class | Algorithm |
|---|---|
| `SPX_Tester2D` | Naive uniform grid |
| `SPX_SLIC2D` | scikit-image SLIC |
| `SPX_Felzenszwalb2D` | scikit-image Felzenszwalb |

### `core/models/timed_annotator.py` — `TimedAnnotatorModel`

Stateful per-session annotation recorder.  Cached once by `ModelRegistry` so
switching away from `TimedAnnotatorFamily` and back preserves the log.

Key responsibilities:

| Method / attribute | Purpose |
|---|---|
| `on_segment_created(segment_id, seg_name, …)` | Appends a `'segment'` log entry with timestamp |
| `on_point_confirmed(segment_id, ras, cp_id, is_negative, …)` | Appends a `'point'` entry; places a mirror control point |
| `on_point_undone(cp_id)` | Removes log entry + mirror point for the given prompt cp_id |
| `sync_visibility(…)` | Matches mirror-node display visibility to the widget's checkbox states |
| `export_data()` | Returns nested per-segment dict (`{"segments": {…}}`) with versioned coord/timestamp lists |
| `on_export(widget)` / `on_import(widget)` | File dialog → JSON write / read |
| `_mirror_to_node(segment_id, ras)` | Lazily creates a per-segment `vtkMRMLMarkupsFiducialNode`; returns the new control-point ID |
| `_on_mirror_point_moved` | Observer: appends a new version when a mirror point is dragged |
| `_on_mirror_point_removed` | Observer: marks point deleted and cleans the log when manually removed from the scene |
| `_PALETTE` | 8-color RGB cycle assigned per segment in creation order |

### `core/utils.py` — shared helpers

| Symbol | Purpose |
|---|---|
| `VIEW_TO_AXIS` | `{"Red": 0, "Green": 1, "Yellow": 2}` |
| `AXIS_TO_XY_COLS` | Maps axis index to the two IJK column indices for the 2-D slice plane |
| `AXIS_TO_IJK_COMPONENT` | Maps axis index to the IJK component that gives the slice number |
| `ras_to_ijk_3d(mat, ras)` | Converts a single RAS point to nearest-integer IJK indices |
| `ras_to_ijk_2d(mat, points, axis, slice_index)` | Projects RAS points to 2-D slice coords; filters by slice |
| `get_slice_from_volume(vol, axis, idx)` | Returns a 2-D view into the 3-D array |
| `write_slice_to_volume(vol, slice, axis, idx)` | Writes a 2-D slice in-place |
| `next_segment_name(existing)` | Returns the lowest `Segment_N` name not in *existing* |
| `parse_user_parameters(text)` | Parses dict-literal or `key=val` param string into a dict |
| `apply_window_level(img, window, level)` | Clips to W/L range and normalises to uint8; no-op if None |
| `select_spx_labels(labels, mask2d, neg_points)` | SPX regions overlapping mask minus neg-point regions |
| `labels_at_points(points, labels)` | Set of label values under given (x, y) points |
| `spx_boundary_mask(labels)` | uint8 mask that is 1 at superpixel boundaries |
| `extract_connected_component(mask, point_xy)` | 4-connected region of *mask* containing *point_xy* |
| `partition_prompt_points(records, preview_ids)` | Splits control points into confirmed / preview lists |
| `collect_confirmed_points(records, preview_ids)` | Confirmed points only (thin wrapper) |
| `collect_preview_points(records, preview_ids)` | Preview-cursor points only (thin wrapper) |
| `call_if_exists(obj, method, *args, **kwargs)` | Calls `obj.method(...)` if the method exists |
| `POSITION_UNDEFINED / POSITION_PREVIEW / POSITION_DEFINED` | Control-point status constants |

---

## Mutual exclusion and handler lifecycle

### One active handler at a time

`widget._active_handler` is the single source of truth for the currently active input handler.  Every `attach()` implementation calls `_detach_current_tool_if_exists(widget)` as its first step, which:

1. Flushes any pending stroke and calls `.detach()` on the previous handler.
2. **Unified pre-attach guard** — if a volume is selected but no segment exists yet, calls `widget.onAddSegment()` before any Slicer element (placement node, effect, mouse filter) is installed.  Single check shared by all handler types.

Two signals switch to `PointHandler`:

| Trigger | Path |
|---|---|
| User clicks a place-widget button | `activeMarkupsFiducialPlaceModeChanged(True)` → `_onPlaceModeChanged` |
| Slicer global toolbar enters Place mode | VTK `InteractionModeChangedEvent` → `_onInteractionModeChanged` |

Both handlers return immediately when `ctrl.is_paused` is True, so programmatic `setCurrentNode` calls (e.g. inside `updateGUIFromParameterNode` or `clearPrompts`) never spuriously activate `PointHandler`.

#### `onSegmentChanged` deferred-signal guard

`qMRMLSegmentSelectorWidget` emits a deferred `currentSegmentChanged` via an internal QTimer after `blockSignals(False)` — by then `creating_segment` is already False.  `onSegmentChanged` stores the last processed segment ID in `_acknowledged_segment_id` and returns immediately on a duplicate, preventing `clearPrompts()` from wiping `_history` and replacing prompt nodes on a spurious re-fire.

#### `onBrushToggled` neutral-off behavior

Toggling the brush button **off** only calls `handler.detach()` — it does **not** re-attach `PointHandler`.  The widget is left in a handler-neutral state.  Previously, toggling off unconditionally called `PointHandler().attach(self)`, which spuriously re-entered persistent point-placement mode.

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

After `_activate_effect()` returns, `StrokeHandler.attach()` checks:

```python
if widget._active_handler is not self:
    return   # superseded — bail out
```

This is defensive code.  In normal operation `_detach_current_tool_if_exists` (called before `widget._active_handler = self`) already ensures a segment exists, so `_activate_effect` itself does not call `onAddSegment`.  The guard prevents a stale `attach()` from installing a duplicate mouse filter or effect callback in the unlikely event that a signal inside `_activate_effect` re-enters `onAddSegment` and replaces `_active_handler`.

---

## Segment visibility

Two independent controls live in `SegmentHumanBody.py`:

| Checkbox | Hotkey | Default | Scope |
|---|---|---|---|
| `showCurrentSegmentCheckBox` ("Show Current Seg") | `V` | checked | Segment being edited |
| `showSegmentsCheckBox` ("Show Saved Segments") | — | unchecked | All other segments |

`_apply_saved_segments_visibility(exclude)` sets `dn.SetSegmentVisibility(sid, _saved_segments_visible)` for every segment except `exclude`.  Called on checkbox toggle, segment switch, and segmentation node change.  On every segment switch the incoming segment is always forced visible and `showCurrentSegmentCheckBox` resets to checked.

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
Pop from `_history` → `logic.reverse_change()` → `tracker.reverse_delta()` → mask pushed to Slicer → entry pushed to `_redo_stack`.  
For `'point'` entries: remove control point first; recreate node if now empty (resets ID counter).

**Redo** (`Ctrl+Shift+Z`):  
Pop from `_redo_stack` → `logic.forward_change()` → `tracker.forward_delta()` → mask pushed to Slicer → entry pushed back to `_history`.  
Any new action (brush, point, expand) clears `_redo_stack`.

**Add Segment** (button, or auto-triggered by any handler attaching to an empty segmentation via the unified guard in `_detach_current_tool_if_exists`):  
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
| `Testing/Python/SegmentHumanBodyTest.py` integration tests | `Slicer.exe --no-main-window --python-script run_slicer_tests.py` (headless) or `Slicer.exe --python-script run_slicer_tests.py` (full GUI) | Yes |

### Pure-Python unit tests (`tests/`)

| File | Covers |
|---|---|
| `test_deps.py` | `DependencyCheck` package/file probing, version comparison, caching, SPX model integration |
| `test_families.py` | SPX label cache hit/miss, `on_expand` behaviour, registry lookup |
| `test_undo_widget.py` | unified undo stack entry format, LIFO ordering, clear semantics, snapshot integrity |
| `test_registry.py` | model registry caching and factory lookup |
| `test_spx_models.py` | individual SPX model algorithms |
| `test_utils.py` | slice read/write helpers, coordinate helpers, window/level, point-collection helpers |

### Slicer-native integration tests (`Testing/Python/SegmentHumanBodyTest.py`)

| Class | Covers |
|---|---|
| `SegmentHumanBodyLogicTest` | `expandSegWithSPX`, `ras_to_ijk_3d`, `getAxisAndSlice`, logic tracker reuse |
| `TrackerUndoTest` | `write_slice` / `reverse_delta` round-trips, bbox efficiency, no-op writes, LIFO undo |
| `UnifiedHistoryTest` | expand returns `MaskChange`; `reverse_change` restores state; LIFO ordering |
| `MouseFilterTest` | `_SliceViewMouseFilter` return value, callback routing, exception safety |
| `AddSegmentHandlerTest` | `onAddSegment` cache/detach/create/restore; detach order; full regression (brush → add segment → place deactivates brush); `StrokeHandler.attach()` supersession guard; unified pre-attach guard (all three handler types auto-create segment on empty segmentation) |
| `BrushStrokeUndoTest` | `commit_stroke` records MaskChange; `onUndo` restores slice; two-stroke LIFO; `EraseHandler._should_track` (no-op vs real erase) |
| `PointPlacementUndoTest` | `_onPointConfirmed` paints correct SPX region (pos/neg); undo removes point + restores mask; two-point LIFO; off-slice point ignored |
| `ManualPointDeletionTest` | `_onPointRemoved` reverses mask and removes history entry; suppressed while paused |
| `MixedActionUndoTest` | brush + point in one session; LIFO ordering across action types; both orderings tested |
| `SegmentVisibilityTest` | `onToggleCurrentSegment` / `onToggleSavedSegments` — hide/show individual and saved segments |
| `WindowLevelTest` | `apply_window_level` normalisation, passthrough when W/L is None, volume not modified |
| `SampleDataWorkflowTest` | end-to-end brush/undo/expand/point cycle on a real CT volume |
| `SPXBrushToggleBugTest` | regression: toggling brush off must not re-enter point-placement mode |
| `SPXSpuriousSegmentChangeBugTest` | regression: deferred `currentSegmentChanged` must not clear `_history` or replace prompt nodes |

See `CLAUDE.md` for exact run commands.  Test runner scripts: `run_slicer_tests.py` (Slicer-native) and `run_tests.py` (standalone PythonSlicer.exe, for debugging only — not the authoritative runner).
