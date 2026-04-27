# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Interpreter

**Always use the Slicer-bundled Python** — there is no standard Python in PATH:

```
C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\bin\PythonSlicer.exe
```

## Running Tests

There are two test layers with different runners.

### Pure-Python unit tests (`tests/`)

Run entirely outside 3D Slicer — no Slicer imports needed. `conftest.py` adds the package root to `sys.path` automatically.

```bash
cd SegmentHumanBody

# All
"C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\bin\PythonSlicer.exe" -m pytest tests/ -v

# Single file
"..." -m pytest tests/test_families.py -v

# Single test
"..." -m pytest tests/test_families.py::TestSPXModelFamily::test_cache_hit_skips_forward -v

# With coverage
"..." -m pytest tests/ --cov=core --cov-report=term-missing
```

Test files:
- `test_families.py` — SPX label cache hit/miss, `on_expand` behaviour, registry lookup
- `test_undo_widget.py` — unified undo stack entry format, LIFO ordering, clear semantics, snapshot integrity
- `test_registry.py` — model registry caching and factory lookup
- `test_spx_models.py` — individual SPX model algorithms
- `test_utils.py` — slice read/write helpers, coordinate helpers

### Slicer-native integration tests (`SegmentHumanBodyTest`)

Defined as `class SegmentHumanBodyTest(ScriptedLoadableModuleTest)` at the bottom of `SegmentHumanBody.py`. These run **inside** a Slicer process and exercise MRML scene interactions (`arrayFromSegmentBinaryLabelmap`, `updateSegmentBinaryLabelmapFromArray`, segmentation node lifecycle, coordinate conversion, and undo).

```bash
# From the command line (requires a build with Slicer in PATH):
PythonSlicer.exe --testing --run SegmentHumanBodyTest
```

Or from inside Slicer: **Developer Tools ▸ Run Unittests**, select `SegmentHumanBodyTest`.

Covered by the integration tests:
- `expandSegWithSPX` selects matched SPX labels and leaves other slices untouched
- `expandSegWithSPX` with `neg_points` subtracts the corresponding SPX labels
- `undo()` restores the pre-expand slice state
- `commit_point` writes pos/neg superpixel selections through the tracker
- `_ras_to_ijk` maps the RAS centre of a volume to the middle voxel
- `onAddSegment` cache/detach/create/restore handler lifecycle
- `StrokeHandler.attach()` supersession guard when `_activate_effect` triggers `onAddSegment`
- Full regression: brush active → Add Segment → click place deactivates brush
- `commit_stroke` records a MaskChange; `onUndo` restores the pre-stroke slice
- Brush LIFO: two strokes on different slices undo in reverse order
- `EraseHandler._should_track`: no-op erase returns None; pixel-removing erase is tracked
- `_onPointConfirmed` with a synthetic SPX family paints the correct superpixel region
- Positive / negative point placement maps the correct label to union / subtract
- Point undo removes the control point from the markup node AND reverses the mask
- Point on a different slice than the current view is silently ignored
- `_onPointRemoved` (manual deletion) reverses the mask without `Ctrl+Z`
- `_onPointRemoved` is suppressed while `ctrl.is_paused`
- Mixed LIFO: brush + point session; each Ctrl+Z pops exactly its own action type

## Architecture

This is a **3D Slicer scripted module**. The entry point (`SegmentHumanBody.py`) is loaded by Slicer and must not be run standalone — it imports `qt`, `vtk`, and `slicer` which only exist inside Slicer's Python.

### Key Layers

```
SegmentHumanBody.py          ← Slicer module entry point: SegmentHumanBodyWidget + Test runner
core/
  _logic.py                  ← SegmentHumanBodyLogic: commit_point, commit_stroke, SPX expansion, prompt nodes, W/L
  _state.py                  ← WidgetState: pause/resume, tool-mode flags
  _input.py                  ← InputHandler hierarchy: StrokeHandler, BrushHandler, EraseHandler, PointHandler
  _tracker.py                ← SegmentTracker: write_slice, reverse_delta, MaskChange
  modelFamilies.py           ← BaseModelFamily, SPXModelFamily, FAMILY_REGISTRY
  modelRegistry.py           ← Lazy-instantiating session cache (ModelRegistry)
  models/spx.py              ← Concrete SPX algorithms (SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D)
  utils.py                   ← get_slice_from_volume, write_slice_to_volume, select_spx_labels, call_if_exists
```

### Single Write Path (Abstract Factory)

All three interactive tools write to the mask through the same `SegmentTracker.write_slice()` path via the `InputHandler` hierarchy:

```
InputHandler
└── StrokeHandler
    ├── BrushHandler   → commit_stroke() → tracker.write_slice()   source='brush'
    └── EraseHandler   → commit_stroke() → tracker.write_slice()   source='erase'
PointHandler           (lifecycle only — no business logic)
```

`_onPointConfirmed` calls `logic.commit_point()` directly: it runs the SPX model via `modelFamily.on_expand()` (using its label cache), finds the superpixel at the click position, and unions (positive) or subtracts (negative) those pixels from the current slice through `tracker.write_slice()`. The resulting `MaskChange` is stored in `_history` synchronously — no async timer needed.

### Mutual Exclusion

Only one handler is active at a time, tracked by `widget._active_handler`. Every `attach()` calls `_detach_current_tool_if_exists(widget)` first, which flushes and detaches the previous handler before the new one activates. Two triggers switch to `PointHandler`:
- `_onPlaceModeChanged(active=True)` — Qt signal from the place widgets (direct path)
- `_onInteractionModeChanged` — VTK observer fallback for Slicer's global toolbar

Both triggers are guarded by `ctrl.is_paused` and return immediately when paused, so programmatic `setCurrentNode` calls inside `updateGUIFromParameterNode` (which run inside `ctrl.pause()`) never spuriously activate `PointHandler`.

#### Segment creation lifecycle (`onAddSegment`)

`clearPrompts()` (called via `onSegmentChanged` → `setCurrentSegmentID`) assigns `_active_handler = PointHandler()` directly without going through the detach lifecycle. To prevent it from orphaning an active stroke handler, `onAddSegment` follows a cache/detach/create/restore sequence:

1. **Cache** — remember the active `StrokeHandler` class (Brush/Erase), if any
2. **Detach** — call `.detach()` on the current handler so it is cleanly removed before creation
3. **Create** — add the empty segment; `onSegmentChanged` → `clearPrompts()` sets `_active_handler = PointHandler()`
4. **Restore** (in `finally`) — if a stroke handler was cached, instantiate and `attach()` it fresh

The restore runs in `finally` so it also executes on early return (e.g. no volume selected), ensuring the handler is always left in a consistent state.

#### `StrokeHandler.attach()` supersession guard

`_activate_effect()` calls `onAddSegment()` when the segmentation has 0 segments (auto-creates the first segment). Because `onAddSegment`'s restore path calls `attach()` on a **new** handler instance, `_active_handler` will no longer point to `self` when `_activate_effect()` returns. The guard:

```python
if widget._active_handler is not self:
    return
```

prevents the original (now superseded) `attach()` from installing a dangling mouse filter and effect callback.

### SPX Label Cache

`SPXModelFamily` caches the superpixel label map inside `on_expand()` using `img.ctypes.data` (the buffer pointer into the VTK volume array) as part of the key — O(1) lookup with no data copy. The cache is invalidated when params change, the image buffer changes, or `confirm_model()` is called.  Both `commit_point()` and the Expand action use this cache.

### Model Family Pattern

UI button visibility is driven entirely by `VISIBLE_BUTTONS` on the active family:

```python
# In Widget.updateUIVisibility():
for name in _BUTTON_NAMES:
    widget.setVisible(name in self.modelFamily.VISIBLE_BUTTONS)
```

Adding a button to a family = add its widget name to `VISIBLE_BUTTONS`.

### Segment Visibility

Two independent visibility controls, each with their own checkbox and state variable:

| Widget | Hotkey | State var | Default | Controls |
|---|---|---|---|---|
| `showCurrentSegmentCheckBox` | `V` | `_current_segment_visible` | `True` | The segment currently being edited |
| `showSegmentsCheckBox` | — | `_saved_segments_visible` | `False` | All other (saved) segments |

`_apply_saved_segments_visibility(exclude=segmentID)` iterates every segment in the segmentation node and calls `dn.SetSegmentVisibility(sid, _saved_segments_visible)` for all except `exclude`. It is called from:
- `onToggleSavedSegments` (checkbox / direct call)
- `onSegmentChanged` (segment switch — hides the previous segment if saved-segments are off)
- `updateGUIFromParameterNode` (segmentation node switch)

On segment switch (`onSegmentChanged`) and on segmentation node change (`updateGUIFromParameterNode`), the incoming current segment is always made visible and `_current_segment_visible` is reset to `True`, so `showCurrentSegmentCheckBox` snaps back to checked.

### Coordinate System

- Prompt points come from Slicer in **RAS** space
- They are converted to **IJK** (voxel) space via `ras_to_ijk()` before passing to models
- Slice extraction: `axis=0` → Red (axial), `axis=1` → Green (coronal), `axis=2` → Yellow (sagittal)
- SPX 2-D point convention: `[x, y]` maps to `labels[y, x]` (row = y, col = x)

### Widget `__init__` attribute groups

```python
# Core state
self.logic           = SegmentHumanBodyLogic()
self.ctrl            = WidgetState(self)
self._parameterNode  = None
self.modelFamily     = None
self.currentViewName = None

# Undo history — entries: ['brush'|'erase'|'expand', MaskChange]
#                          ['point', MaskChange, node, cp_id]
self._history = []

# Active input handler
self._active_handler = None

# Keyboard shortcuts (assigned in setup())
self._undo_shortcut         = None   # Ctrl+Z
self._expand_shortcut       = None   # E
self._spx_boundary_shortcut = None   # Q
self._segments_shortcut     = None   # V

# SPX boundary overlay
self._spx_boundary_node    = None
self._spx_boundary_visible = False
self._spx_boundary_view    = None

# Segment visibility
self._saved_segments_visible  = False   # saved segments checkbox
self._current_segment_visible = True    # current segment / V hotkey
```

### Widget method sections (in order)

`# Lifecycle` → `# UI` → `# Signals & Observers` → `# Parameter Node` → `# Model selection` → `# Segment management` → `# Interaction mode` → `# Point events` → `# Brush tool` → `# Window / Level` → `# Expand (E)` → `# Undo (Ctrl+Z)` → `# SPX Boundary Overlay (Q)` → `# Segment Visibility`

### Undo System

All undo actions follow one path: pop a `['type', MaskChange, ...]` entry from `_history`, call `logic.reverse_change()` which calls `tracker.reverse_delta()`.

| Entry type | Extra fields | What undo does |
|---|---|---|
| `'brush'` | change | reverse delta |
| `'erase'` | change | reverse delta |
| `'expand'` | change | reverse delta |
| `'point'` | change, node, cp_id | remove control point + reverse delta; recreate node if now empty (resets ID counter) |

Manual point deletion (not via Ctrl+Z) is handled by `_onPointRemoved`, which scans `_history` for the matching cp_id and calls `reverse_change()`.

### Adding a New SPX Model

1. Implement the class in `core/models/spx.py` with `PARAM_HINT`, optional `DOC_URL`, and `forward(**kwargs)` that pops `img` and returns an integer label map.
2. Register it in `core/modelRegistry.py` `_MODEL_FACTORIES` dict.
3. Add the display name → registry key mapping to `SPXModelFamily.MODEL_MAP` in `core/modelFamilies.py`.
4. `core/models/spx.py` is already in `CMakeLists.txt`; new files must be added there.

### Adding a New Model Family

1. Subclass `BaseModelFamily` in `core/modelFamilies.py`.
2. Populate `VISIBLE_BUTTONS` with the widget names that should appear for this family.
3. Add the display name → class mapping to `FAMILY_REGISTRY` at the bottom of `modelFamilies.py`.
