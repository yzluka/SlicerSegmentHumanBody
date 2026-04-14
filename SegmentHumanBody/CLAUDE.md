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
- `test_families.py` — model family logic (SPX cache, base-mask additive/subtractive mode, neg/pos point handling, propagate)
- `test_undo_stack.py` — per-segment LIFO undo stack
- `test_registry.py` — model registry caching and factory lookup
- `test_spx_models.py` — individual SPX model algorithms
- `test_utils.py` — slice read/write helpers

### Slicer-native integration tests (`SegmentHumanBodyTest`)

Defined as `class SegmentHumanBodyTest(ScriptedLoadableModuleTest)` at the bottom of `SegmentHumanBody.py`. These run **inside** a Slicer process and exercise MRML scene interactions (`arrayFromSegmentBinaryLabelmap`, `updateSegmentBinaryLabelmapFromArray`, segmentation node lifecycle, coordinate conversion, undo, and the interactive base mask).

```bash
# From the command line (requires a build with Slicer in PATH):
PythonSlicer.exe --testing --run SegmentHumanBodyTest
```

Or from inside Slicer: **Developer Tools ▸ Run Unittests**, select `SegmentHumanBodyTest`.

Covered by the integration tests:
- `applyResult` auto-creates segment and writes to the correct slice
- `applyResult` reuses `_working_mask` across frames (no re-allocation)
- `expandSegWithSPX` selects matched SPX labels and leaves other slices untouched
- `expandSegWithSPX` with `neg_points` subtracts the corresponding SPX labels
- `undo()` restores the pre-expand slice state
- `on_enter_interactive` snapshots `_interactive_base_mask` from the current labelmap
- `reset_render_state` clears `_interactive_base_mask`
- `_ras_to_ijk` maps the RAS centre of a volume to the middle voxel

## Architecture

This is a **3D Slicer scripted module**. The entry point (`SegmentHumanBody.py`) is loaded by Slicer and must not be run standalone — it imports `qt`, `vtk`, and `slicer` which only exist inside Slicer's Python.

### Key Layers

```
SegmentHumanBody.py          ← Slicer module: Widget + Logic
core/
  modelFamilies.py           ← BaseModelFamily, SAMFamily, SPXModelFamily, AutoModelFamily
  modelRegistry.py           ← Lazy-instantiating session cache (ModelRegistry)
  models/spx.py              ← Concrete SPX algorithms (SPX_Tester2D, SPX_SLIC2D, SPX_Felzenszwalb2D)
  undoStack.py               ← Per-segment LIFO history of 2D slice snapshots
  utils.py                   ← get_slice_from_volume, write_slice_to_volume, call_if_exists
```

### Model Family Pattern

UI button visibility is driven entirely by method presence on the active family instance:

```python
# In Widget.updateUIVisibility():
widget.setVisible(hasattr(self.modelFamily, method_name))
```

Adding a button means: add it to the mapping list AND add the method to the appropriate family class.

### SPX Interactive Loop

```
SegmentationRenderer (QTimer 100ms)
  → Logic.onRender()
      → render_key check (skip if points/slice/params unchanged)
      → get_slice_from_volume (numpy view, no copy)
      → SPXModelFamily.onRender()
          → _make_cache_key (img.ctypes.data pointer — O(1), no copy)
          → model.forward() only on cache miss
          → apply pos/neg label selections + base_mask
      → Logic.applyResult()
          → _working_mask (persistent 3D array, updated 2D slice in-place)
          → updateSegmentBinaryLabelmapFromArray
```

**`_interactive_base_mask`**: Snapshot of the segment's 3D labelmap taken when entering interactive mode. Each render computes `result = (base_slice | pos_selections) & ~neg_selections`. Removing a pos point reverts that region to base state.

**`_working_mask`**: Persistent 3D numpy array kept in sync with Slicer to avoid a full `arrayFromSegmentBinaryLabelmap` call every 100ms. Keyed by `(segNodeID, segmentID)`.

### SPX Label Cache

`SPXModelFamily` caches the superpixel label map using `img.ctypes.data` (the buffer pointer into the VTK volume array) as part of the key — O(1) lookup with no data copy. The cache is invalidated when params change, the image buffer changes, or `confirm_model()` is called.

### Coordinate System

- Prompt points come from Slicer in **RAS** space
- They are converted to **IJK** (voxel) space via `_ras_to_ijk()` before passing to models
- Slice extraction: `axis=0` → Red (axial), `axis=1` → Green (coronal), `axis=2` → Yellow (sagittal)
- SPX point convention: `[x, y]` maps to `labels[y, x]` (row = y, col = x)

### Undo System

- **Interactive mode** (renderer running): Ctrl+Z removes the last added prompt point from `_interactive_point_stack` (node, controlPointID pairs). The render loop redraws automatically.
- **Non-interactive mode**: Ctrl+Z pops a 2D slice snapshot from `UndoStack` and restores it. Snapshots are pushed in `expandSegWithSPX` before writing.

### Adding a New SPX Model

1. Implement the class in `core/models/spx.py` with `PARAM_HINT`, optional `DOC_URL`, and `forward(**kwargs)` that pops `img` and returns an integer label map.
2. Register it in `core/modelRegistry.py` `_MODEL_FACTORIES` dict.
3. Add the display name → registry key mapping to `SPXModelFamily.MODEL_MAP` in `core/modelFamilies.py`.
4. Add `core/models/spx.py` is already in `CMakeLists.txt`; new files must be added there.

### Adding a New Model Family

1. Subclass `BaseModelFamily` in `core/modelFamilies.py`.
2. Define only the methods that correspond to buttons you want visible (the visibility system uses `hasattr`).
3. Add the display name → class mapping to `SegmentHumanBodyWidget.model_classes`.
