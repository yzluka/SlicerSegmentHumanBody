# Project Structure

This repository is a 3D Slicer scripted module. It is not a standalone Python
application: the module runs inside Slicer, which provides the `qt`, `vtk`, and
`slicer` Python bindings.

The current active branch is `feature/native-editor-wrapper`. Its design goal is
to keep Slicer's native Segment Editor as the editing engine, while adding a
mouse-centered recording/export layer for annotation-process
analysis.

## Repository Layout

```text
SlicerSegmentHumanBody/
|-- README.md
|-- TODO.md
|-- PROJECT_STRUCTURE.md
|-- run_slicer_tests.py
|-- SegmentHumanBody/
|   |-- SegmentHumanBody.py
|   |-- CMakeLists.txt
|   |-- CLAUDE.md
|   |-- SPEC.md
|   |-- TOFIX.md
|   |-- Resources/UI/SegmentHumanBody.ui
|   |-- core/
|   |   |-- _input.py
|   |   |-- _logic.py
|   |   |-- _state.py
|   |   |-- _tracker.py
|   |   |-- _deps.py
|   |   |-- _point_log.py
|   |   |-- _audio_recorder.py
|   |   |-- _mouse_recorder.py
|   |   |-- TimeLogInterpreter.py
|   |   |-- TimeLogSummarizer.py
|   |   |-- utils.py
|   |   |-- modelFamilies.py
|   |   |-- modelRegistry.py
|   |   |-- models/default.py
|   |   |-- models/spx.py
|   |   `-- models/timed_annotator.py
|   |-- tests/
|   |   |-- conftest.py
|   |   |-- test_families.py
|   |   |-- test_registry.py
|   |   |-- test_utils.py
|   |   |-- test_deps.py
|   |   |-- test_spx_models.py
|   |   |-- test_mouse_recorder.py
|   |   |-- test_time_log_summarizer.py
|   |   |-- test_segment_lifecycle_recording.py
|   |   |-- test_navigation_shortcuts.py
|   |   |-- test_audio_recorder.py
|   |   `-- test_undo_widget.py
|   `-- Testing/Python/SegmentHumanBodyTest.py
`-- SegmentHumanBody/models/
    |-- sam/
    |-- sam2_annotation_tool/
    |-- segment_any_muscle/
    |-- breast_model/
    `-- ct_segmentation/
```

## Active Runtime Architecture

### `SegmentHumanBody.py`

The Slicer module entry point. It owns:

- Qt UI setup and signal wiring.
- Volume, segmentation, segment, visibility, window/level, and prompt-node UI.
- The current active input handler (`_active_handler`).
- The default model-family placeholder: `Default` / `Identity`.
- The process recorder singleton (`MouseEventRecorder`).
- Recording/export button handlers.
- Module-scoped keyboard shortcuts:
  - `A` / `W`: next / previous loaded scalar volume sequence.
  - `Z` / `C`: previous / next segment.
  - `Q`: show/hide current segment.
  - `S`: show/hide saved/other segments.

Brush and erase delegate to Slicer's native Segment Editor effects. Undo and
redo delegate to Slicer's native Segment Editor undo stack.

Volume switching is intentionally independent from segmentation selection.
Radiology folders may contain several sequences but only one intended
segmentation, so `A`/`W` update the source volume and slice views without
auto-matching, clearing, or switching segmentation nodes. If the target volume
does not match the selected segmentation's voxel grid shape, spacing, or
orientation, the user is asked whether to create a new empty segmentation, keep
the current segmentation, or cancel the switch. Origin differences are ignored
for this compatibility check. The widget observes scene volume imports and
normalizes compatible zero-origin derived volumes by copying the first matching
non-zero origin in the scene. The active workflow assumes one dragged folder is
one patient set: all loaded scalar volumes are checked as one group, and
inconsistent shape/spacing/orientation shows one informational warning after
the import stream settles, with geometry-group statistics and filenames, while
still allowing loading and switching. If a
segment is explicitly created when no segmentation exists, a
generic segmentation node is created for the active volume. Recording stores the
explicit selected volume, segmentation, and segment identities instead of
relying on naming conventions.

`A`/`W` and `Z`/`C` wrap around at the ends. Volume switching preserves the
current native Slicer slice view state and avoids refitting the views.
Recordings store all loaded scalar volume sequences in metadata, and volume
selector changes are interpreted as compact semantic `volume_change` events.
The `Clear Loaded Volumes` button removes scalar volume nodes only; segmentation
nodes remain manually controlled.
Re-imported scalar volume files replace older nodes loaded from the same storage
path and use the full storage filename, including suffix such as `.nii.gz`,
instead of accumulating `_1` duplicates.

### `core/_input.py`

Defines the handler wrappers:

| Handler | Wrapped Slicer tool | Attach behavior | Detach behavior |
|---|---|---|---|
| `BrushHandler` | Segment Editor `Paint` | Ensures segment, activates Paint | Deactivates Paint and unchecks tool buttons |
| `EraseHandler` | Segment Editor `Erase` | Ensures segment, activates Erase | Deactivates Erase and unchecks tool buttons |
| `PointHandler` | `qSlicerSimpleMarkupsWidget` place mode | Ensures segment/prompt nodes, enables place mode | Disables prompt placement |

Only one handler should be active at a time. A handler is considered a wrapper:
detaching it must also disable the Slicer tool it wraps.

### `core/_mouse_recorder.py`, `core/TimeLogInterpreter.py`, and `core/TimeLogSummarizer.py`

Three-stage recording pipeline:

1. **`_mouse_recorder.py`** captures raw device events and writes `*_raw.json`.
2. **`TimeLogInterpreter.py`** converts `*_raw.json` → compact semantic `annotation_process` log (`*.json`).
3. **`TimeLogSummarizer.py`** converts the `annotation_process` log → a human-readable `annotation_summary` with higher-level activity spans.

`TimeLogSummarizer` groups consecutive low-level events into named spans
(`stroke`, `click`, `volume_navigation`, `slice_navigation`, `point_click_place`,
`point_drag`, etc.) and maintains running state (volume/tool/segment/view/slice)
so each span carries full context. Output has both a structured `spans` array
and a `text` array (one readable line per span).

Both `TimeLogInterpreter` and `TimeLogSummarizer` are standalone — no Slicer
dependency — and can run offline from the saved JSON files.

Mouse-centered recorder plus offline raw-log interpreter.

Recording is intentionally not a general UI macro. It captures:

- Sequential `event_id` values for fast retrieval and stable event references.
- Mouse trajectory inside the active volume only.
- Slice-view Qt events plus high-priority, non-consuming VTK interactor events,
  so native Segment Editor brush/erase input is visible to the recorder before
  Paint/Erase can consume it.
- Raw input export is the source of truth. Compact interpreted export is
  regenerated from `_raw.json` by `TimeLogInterpreter`.
- Mouse raw entries keep original input fields: view, slice index, VTK device
  XY, optional global screen XY, pressed state, and wheel delta when relevant.
- Raw source entries also keep tool selection, segment lifecycle/selection,
  model selection, brush parameter changes, and markups point events needed for
  offline interpretation outside Slicer.
- Metadata is not an event; exported event IDs start at 1 for process events.
- Interpreted events use IJK. Mouse IJK is derived from raw XY plus stored
  per-view `xy_to_ijk`; point IJK is derived from raw markups world position
  plus recorded volume metadata.
- Mouse status: `move`, `press`, `release`, or `view`.
- Trajectory kind/role: held-button edit paths are `annotation_move` /
  `annotation_trajectory`; released-button hover paths are
  `non_annotation_move` / `visualization_trajectory`; wheel/view changes use
  `view_change`.
- Active handler/tool: `brush`, `erase`, `point`, or `None`.
- Event-specific parameters only, such as brush radius for brush/erase.
- Initial Red/Green/Yellow visualization state in metadata, with later
  snapshots only for explicit view-change events and without repeated
  slice-view dimensions.
- Semantic point additions from markups-node events.
- Point placement from markups-node events: one defined-location
  `point_placed` verdict with point ID/name, with press-release drift treated
  as `non_annotation_move`.
- Existing-point relocation from markups-node events: start is
  `point_drag_start`/`grab`, movement is sampled `non_annotation_move`
  visualization trajectory, and release is `point_replaced`/`replace`.
- Control-point deletion from markups-node events: `point_removed` with the
  last cached point location.
- Segmentation-changing actions such as segment remove/rename and undo/redo.
- Segment switches record raw `segment_selected`; active segment context is
  still carried by boundary and trajectory payloads.
- Brush/erase drag samples or releases infer a missing `press` boundary if
  Slicer does not deliver the initial press event.
- Brush/erase press, release, held-button movement, and released-button
  movement are listener-recordable inside volume views.
- Slice-view movement such as wheel/scroll as `view_changed` visual trajectory
  events, including the recorded Red/Green/Yellow visual state.
- Initial Red/Green/Yellow slice visual states in one `metadata` record.
- Slice-view mouse coordinates come from the single VTK interactor listener
  (`GetEventPosition()` device XY) and are interpreted through the same
  DataProbe-style XY-to-IJK path used for active-volume checks.

Movement is considered by timer at 60 records/sec and thinned by cached
XY-to-IJK scale. Annotation moves target >=0.5 IJK voxels clamped to 1-4 px;
hover moves target >=2 IJK voxels clamped to 2-12 px; 100/250 ms time caps keep
continuity. The compact policy is stored in `metadata.move_thinning`.
Active-volume membership is checked on cached per-view DataProbe-style
XY-to-IJK matrices, so inactive XY is ignored before it accumulates. Export
still recomputes IJK bounds from stored metadata for consistency.

Segment creation, removal, rename, and selection are observed as raw source
events. They are not brush/point annotation trajectories.

### `core/_audio_recorder.py`

Standalone timestamped microphone/audio recorder for future local Whisper
integration. It is not wired into the Slicer UI yet. It lazy-imports
`sounddevice` only when microphone capture starts, writes Whisper-friendly
16 kHz mono PCM WAV chunks by default, and produces chunk metadata with absolute
start/end timestamps plus a JSON manifest helper.

### `core/modelFamilies.py` and `core/modelRegistry.py`

The model-family framework is present but mostly hidden in the UI on this
branch. The current default template is:

| Family | Variant | Model | Behavior |
|---|---|---|---|
| `Default` | `Identity` | `IdentityModel` | Returns input image unchanged |

This provides a small template for future model-family integrations without
changing current native-editor behavior.

SPX code and tests remain as reusable infrastructure, but superpixel-grid
display and any related shortcut are deferred on this branch.

## Tests

### Pure-Python Tests

Run from `SegmentHumanBody/`:

```powershell
PythonSlicer.exe -m pytest tests/ -q
```

These cover model registry/families, utility functions, dependency checks,
mouse-recorder data formatting, time-log summarizer spans, audio recorder
metadata, segment lifecycle recording, and navigation shortcuts — all runnable
outside a full Slicer GUI process.
### Slicer-Native Tests

Run from the repo root:

```powershell
& 'C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe' --no-main-window --python-script D:/SlicerSegmentHumanBody/run_slicer_tests.py
```

Full GUI runs are still needed for behaviors that require live slice views or
interactive markups placement.
