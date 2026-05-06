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
|   |   |-- _mouse_recorder.py
|   |   |-- _point_log.py
|   |   |-- modelFamilies.py
|   |   |-- modelRegistry.py
|   |   |-- models/default.py
|   |   |-- models/spx.py
|   |   `-- models/timed_annotator.py
|   |-- tests/
|   |   |-- test_families.py
|   |   |-- test_mouse_recorder.py
|   |   |-- test_segment_lifecycle_recording.py
|   |   |-- test_registry.py
|   |   |-- test_spx_models.py
|   |   `-- test_utils.py
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

Brush and erase delegate to Slicer's native Segment Editor effects. Undo and
redo delegate to Slicer's native Segment Editor undo stack.

### `core/_input.py`

Defines the handler wrappers:

| Handler | Wrapped Slicer tool | Attach behavior | Detach behavior |
|---|---|---|---|
| `BrushHandler` | Segment Editor `Paint` | Ensures segment, activates Paint | Deactivates Paint and unchecks tool buttons |
| `EraseHandler` | Segment Editor `Erase` | Ensures segment, activates Erase | Deactivates Erase and unchecks tool buttons |
| `PointHandler` | `qSlicerSimpleMarkupsWidget` place mode | Ensures segment/prompt nodes, enables place mode | Disables prompt placement |

Only one handler should be active at a time. A handler is considered a wrapper:
detaching it must also disable the Slicer tool it wraps.

### `core/_mouse_recorder.py`

Mouse-centered process recorder.

Recording is intentionally not a general UI macro. It captures:

- Sequential `event_id` values for fast retrieval and stable event references.
- Mouse trajectory inside the active volume only.
- Slice-view Qt events plus high-priority, non-consuming VTK interactor events,
  so native Segment Editor brush/erase input is visible to the recorder before
  Paint/Erase can consume it.
- Compact export as `{type, metadata, events}` with event `id` and
  `t_ms` relative to recording start.
- Metadata is not an event; exported event IDs start at 1 for process events.
- RAS position relative to the active volume; IJK is derivable from RAS plus
  recorded volume metadata and is not duplicated in every event.
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
  visualization trajectory, and release is `point_placed`/`replace`.
- Control-point deletion from markups-node events: `point_removed` with the
  last cached point location.
- Segmentation-changing actions such as segment remove/rename and undo/redo.
- Segment switches are not standalone events; the active segment is carried by
  boundary, trajectory, and semantic event payloads.
- Brush/erase drag samples or releases infer a missing `press` boundary if
  Slicer does not deliver the initial press event.
- Brush/erase press, release, held-button movement, and released-button
  movement are listener-recordable inside volume views.
- Slice-view movement such as wheel/scroll as `view_changed` visual trajectory
  events, including the recorded Red/Green/Yellow visual state.
- Initial Red/Green/Yellow slice visual states in one `metadata` record.

Movement is sampled by timer at 30 records/sec. Raw movement is resolved into
in-volume RAS samples before classification, unchanged positions are skipped,
and events outside the active volume are ignored until the cursor maps back
inside the volume.

Segment removal and rename are observed on the active `vtkSegmentation` object.
Segment creation is not recorded as a standalone process event.

### `core/modelFamilies.py` and `core/modelRegistry.py`

The model-family framework is present but mostly hidden in the UI on this
branch. The current default template is:

| Family | Variant | Model | Behavior |
|---|---|---|---|
| `Default` | `Identity` | `IdentityModel` | Returns input image unchanged |

This provides a small template for future model-family integrations without
changing current native-editor behavior.

## Tests

### Pure-Python Tests

Run from `SegmentHumanBody/`:

```powershell
PythonSlicer.exe -m pytest tests/ -q
```

These cover model registry/families, utility functions, dependency checks, and
mouse-recorder data formatting that can run outside a full Slicer GUI process.
### Slicer-Native Tests

Run from the repo root:

```powershell
& 'C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe' --no-main-window --python-script D:/SlicerSegmentHumanBody/run_slicer_tests.py
```

Full GUI runs are still needed for behaviors that require live slice views or
interactive markups placement.
