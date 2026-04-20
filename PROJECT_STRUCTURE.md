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
  (`_history`), the active stroke handler, and SPX boundary overlay state.
  Delegates all logic to the classes below.
- **`SegmentHumanBodyTest`** — `ScriptedLoadableModuleTest` integration test runner.

### `core/_logic.py` — `SegmentHumanBodyLogic`

All stateful business logic that does not belong in the widget.  Key
responsibilities:

| Attribute / method | Purpose |
|---|---|
| `_tracker` | `SegmentTracker` instance; mask cache for the current segment |
| `_session_base` | Frozen 3-D snapshot taken on the first prompt; base for SPX renders |
| `_erase_acc` | `{(axis, idx): bool 2D array}` pixels excluded by manual erase strokes |
| `_last_render_key` | Tuple of (points, axis, slice, params); skips render when unchanged |
| `onRender()` | Main render loop: builds render key, extracts slice, calls model family |
| `applyResult()` | Writes model output through `SegmentTracker.write_slice()` |
| `commit_stroke()` | Reconciles a brush/erase stroke into `_tracker` and `_session_base` |
| `reverse_change()` | Reverses a `MaskChange` via `_tracker.reverse_delta()`; syncs `_session_base` and `_erase_acc` |
| `invalidate_render_key()` | Clears `_last_render_key` only — preserves `_session_base` and `_erase_acc` |
| `reset_render_state()` | Clears render key, session snapshot, and erase accumulator |
| `on_expand()` | Runs `expandSegWithSPX` and returns the `MaskChange` for undo |
| `recreate_prompt_node()` | Replaces one markup node to reset its ID counter |
| `recreatePromptNodes()` | Replaces both markup nodes (used by `clearPrompts`) |

### `core/_state.py` — `WidgetState`

Centralises all boolean semaphores to avoid race conditions:

| Flag | Meaning |
|---|---|
| `activating_brush` | True while `StrokeHandler._activate_effect` is running |
| `brush_in_progress` | True between mouse-down and mouse-up during a stroke |
| `creating_segment` | Suppresses `onSegmentChanged` while a new segment is being added |
| `is_paused` | Nestable pause; `request_render` is a no-op while paused |
| `is_rendering` | True while the render callback is executing; guards re-entrancy |

Also owns `request_render()` — the single dispatching point for the render loop.

### `core/_input.py` — input handler hierarchy

```
InputHandler          ← base (attach / detach / flush lifecycle)
└── StrokeHandler     ← owns mouse filter + stroke-before snapshot
    ├── BrushHandler  ← EFFECT='Paint', SOURCE='brush'
    └── EraseHandler  ← EFFECT='Erase', SOURCE='erase'; skips no-op strokes
```

- **`_SliceViewMouseFilter`** — application-level Qt event filter; fires
  `on_press` / `on_release` callbacks on left-button events.
- **`StrokeHandler`** — captures a before-snapshot on mouse-down, commits
  via `logic.commit_stroke()` on mouse-up (through a 0-ms timer so Slicer's
  Paint effect apply() finishes first).  Stores the result in `_history`.

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
├── SAMFamily        ← SAM v1/v2 variants; onRender is a placeholder
└── SPXModelFamily   ← superpixel algorithms; fully implemented
```

UI button visibility is driven entirely by `VISIBLE_BUTTONS` — the widget's
`updateUIVisibility()` shows/hides each managed button based on set membership.
Adding a button to a family = add its widget name to `VISIBLE_BUTTONS`.

`SPXModelFamily.onRender()` implements the SPX formula:

```
result = where(neg_region, 0,
           where(pos_region & ~erase_mask, 1, base_mask))
```

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

## Data flow summary

```
User interaction (click / scroll / brush)
  │
  ├─ Brush / Erase ──► StrokeHandler ──► logic.commit_stroke()
  │                                           │
  │                                    SegmentTracker.write_slice()
  │                                    _session_base sync
  │                                    _erase_acc update
  │
  ├─ Point placed ──► _onPointConfirmed ──► _triggerRender
  │
  └─ Slice scroll ──► slice observer ──► _triggerRender
                                              │
                                        WidgetState.request_render()
                                              │
                                        SegmentHumanBodyLogic.onRender()
                                              │
                                        render_key check (skip if unchanged)
                                              │
                                        SPXModelFamily.onRender()
                                              │
                                        logic.applyResult()
                                              │
                                        SegmentTracker.write_slice()
```

**Undo** (`Ctrl+Z`):  
Pop from `_history` → `logic.reverse_change()` → `tracker.reverse_delta()` +
`_session_base` sync + `_erase_acc` un-accumulate → `invalidate_render_key()` →
trigger render.

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

See `CLAUDE.md` for exact run commands.
