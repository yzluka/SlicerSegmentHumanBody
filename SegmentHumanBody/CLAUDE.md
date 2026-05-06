# CLAUDE.md

Guidance for AI assistants and future maintainers working on this repository.

## Current Branch Direction

This branch (`feature/native-editor-wrapper`) keeps the current UI layout and
uses Slicer's native Segment Editor for editing. Do not reimplement brush,
erase, undo, redo, or markup placement unless there is a clear Slicer limitation.

The main development focus is the mouse-centered record/export
system for annotation-process analysis.

## Development Principles

1. Prefer Slicer native tools.

   - Brush/erase: Segment Editor `Paint` / `Erase`.
   - Undo/redo: Segment Editor undo stack.
   - Prompt points: `qSlicerSimpleMarkupsWidget` and
     `vtkMRMLMarkupsFiducialNode`.
   - Coordinates: Slicer RAS/IJK matrices.

2. Handlers are wrappers.

   `BrushHandler`, `EraseHandler`, and `PointHandler` wrap Slicer tools. If a
   handler detaches, it must also disable the wrapped Slicer tool. Mutual
   exclusion means only one wrapper and one wrapped Slicer tool are active.

3. Recording is mouse-centered, not UI-macro-centered.

   Record:

   - Mouse trajectory in 3D relative to the active volume.
   - Trajectory kind: `annotation_move` for mouse-held edits and
     `non_annotation_move` for released-button hover and point-relocation
     movement that does not directly modify masks.
   - Trajectory role: `annotation_trajectory` for edit paths and
     `visualization_trajectory` for hover/view-only movement.
   - Active handler/tool and handler params.
   - Mouse status (`move`, `press`, `release`, `view`).
   - Visualization state.
   - Mouse-triggered semantic operations.
   - Hotkeys/actions that modify segmentation masks or segment state.

   Do not record general UI-panel motion as annotation process data.

4. Logic remains UI-light.

   Widget code may read UI state and wire Slicer widgets. Logic helpers should
   avoid depending on Qt widgets unless the active branch has already placed the
   behavior in `SegmentHumanBodyWidget`.

## Recording Contract

`core/_mouse_recorder.py` records only events that occur in Red/Green/Yellow
slice views and map inside the active volume.

Current schema is intentionally strict while this branch is under test:

- Saved recordings are compact process logs with top-level `type`, `metadata`,
  and `events`.
- `xy_global` / screen-space coordinates are not exported.
- RAS is the canonical coordinate. 3D IJK can be derived later from each event
  RAS plus recorded volume metadata; do not duplicate IJK into every event.

The recorder is an event-listener-style component. It should first produce a
correct in-volume event stream, then annotate events with
annotative/non-annotative classification. Do not let policy decisions drop raw
press/release/move events inside volume views.

Exactly one `metadata` record is created when recording starts. It caches volume
metadata, the 30 Hz sample rate, the initial Red/Green/Yellow visual state, and
whether recording began with the left mouse button already down.

Metadata is not an exported event. The compact export stores:

- `metadata.start_time`: absolute time for the recording.
- `events[].id`: sequential event reference starting at 1 for process events
  only.
- `events[].t_ms`: event time relative to `metadata.start_time`.
- `events[].ras`: 3D Slicer RAS coordinate when the event has an in-volume
  position.
- `events[].mouse`: `move`, `press`, `release`, or `view`.
- `events[].pressed`: `1` when the left mouse button is down, `0` when
  released.
- `events[].kind`: `annotation_move`, `non_annotation_move`, or `view_change`.
- `events[].role`: `annotation_trajectory` or `visualization_trajectory`.
- `events[].tool`: `brush`, `erase`, `point`, or `None`.
- `events[].brush_mm`: brush/erase radius when relevant.
- Point fields (`point`, `point_index`, `point_action`, `negative`) only when
  relevant.
- View fields (`view_event`, `wheel_delta`, `visual_state`) only for explicit
  view-change events. Initial Red/Green/Yellow state lives in metadata.
  Visual-state snapshots omit repeated slice-view dimensions.

Movement samples the latest in-volume cursor position at 30 records/sec and
skips unchanged positions.

Slice-view capture uses both Qt event filters and high-priority, non-consuming
VTK interactor observers because native Segment Editor Paint/Erase may bypass
one path or abort propagation after handling the event. Avoid doing
unnecessary work in raw mouse callbacks, but do not defer active-volume
membership until the timer: each raw move should resolve XY to RAS immediately
and cache the latest valid in-volume sample with its original timestamp. The
timer only appends the latest cached valid sample. Boundary events should flush
pending movement before appending themselves so event order remains meaningful.
When VTK move capture is available, Qt `MouseMove` capture is disabled to avoid
duplicate high-frequency callbacks.

Brush/erase press is a required boundary event. If Slicer drops the initial
press but a brush/erase drag sample or release is observed, infer a `press`
boundary before the first sampled move/release and set
`payload.boundary_source`.
Brush/erase movement is classified from the mouse button state: held-button
movement is `annotation_move`; released-button movement is
`non_annotation_move`. Both are listener-recordable in volume views.

Point placement is recorded semantically via markups-node events as one
`point_placed` boundary, not as raw press plus release mouse boundaries. A new
point records its `point_placed` verdict on `PointPositionDefinedEvent`, using
the defined control-point location. The later interaction end is used only to
refresh cached metadata and must not duplicate the verdict. Include point ID
and point name/label when available. Minor drift between press and release is
`non_annotation_move` trajectory.

Point relocation is also semantic annotation process data. Record markups
`PointStartInteractionEvent` as `point_drag_start` with `point_action: grab`;
record `PointEndInteractionEvent` as `point_placed` with
`point_action: replace` at the final RAS. Sample `PointModifiedEvent` movement
as `non_annotation_move` instead of creating a new point.
During compact export, suppress raw point-tool mouse press/move/release
companions for an accepted point relocation; those raw mouse RAS values are a
different coordinate stream and are redundant with the markups point stream.
Only start point-drag recording for previously defined control points, not
preview/not-yet-assigned placement points.
Point-drag move/end events must be ignored unless a valid point-drag start was
accepted first; this prevents pre-assignment hover/preview movement from being
misclassified as relocation trajectory.
Point-drag move sampling should be checked before node lookup/RAS extraction so
high-frequency markups `PointModifiedEvent` callbacks stay cheap.
Point deletion records `point_removed` using the last cached point metadata so
the event can still include RAS after Slicer removes the control point.

Segment creation is not recorded as a standalone process event. Segment removal
and segment rename are recorded from active `vtkSegmentation` events:
`segment_removed` carries the last known name, and `segment_renamed` carries
old/new names.

Segment selection changes are not standalone records. The active segment ID is
carried on boundary, trajectory, and semantic payloads.

Guard signal loops explicitly. In particular, parameter-node-to-UI sync uses
`_syncing_parameter_node_to_ui` so selector changes caused by sync do not write
straight back into the parameter node.

## Model Family Template

The current placeholder is:

- Family: `Default`
- Variant: `Identity`
- Model: `IdentityModel`

This is the template for future families. Add new models through
`core/modelRegistry.py`, then expose them through a family in
`core/modelFamilies.py`.

## Python Interpreter

Use Slicer's Python:

```powershell
PythonSlicer.exe
```

Known local Slicer executable:

```powershell
C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe
```

## Tests

Pure-Python tests:

```powershell
cd D:\SlicerSegmentHumanBody\SegmentHumanBody
PythonSlicer.exe -m pytest tests/ -q
```

Slicer-native tests:

```powershell
cd D:\SlicerSegmentHumanBody
& 'C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe' --no-main-window --python-script D:/SlicerSegmentHumanBody/run_slicer_tests.py
```

Some handler/markups tests require a live Slicer Qt runtime and are skipped by
plain `PythonSlicer.exe` when the full Qt/Slicer application is not available.
