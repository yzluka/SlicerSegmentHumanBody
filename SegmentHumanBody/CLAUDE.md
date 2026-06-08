# CLAUDE.md

Guidance for AI assistants and future maintainers working on this repository.

## Current Branch Direction

This branch (`annotation-process-recorder`) uses Slicer's native Segment Editor
as the editing engine and adds a mouse-centered annotation-process recorder for
studying how annotators work.

Do not reimplement brush, erase, undo, redo, or markup placement unless there is
a clear Slicer limitation.

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

5. Keep volume sequence navigation separate from segmentation selection.

   `A` / `W` move through loaded scalar volumes in MRML scene order and update
   the Red/Green/Yellow slice views. They must not auto-match, clear, or switch
   segmentation nodes. Segmentation selection is manual because a folder may
   contain multiple radiology sequences but only one intended segmentation. If
   the selected segmentation voxel grid shape, spacing, or orientation differs
   from the target volume, ask whether to create a new empty segmentation, keep
   the current segmentation, or cancel the switch. Origin differences are
   ignored for this compatibility check. Observe scene volume imports and copy
   the first compatible non-zero origin onto zero-origin derived volumes before
   display. Treat one dragged folder as one patient volume set: all loaded
   scalar volumes should have consistent shape, spacing, and orientation.
   Inconsistent spacing/shape/orientation should show one informational warning
   after the import stream settles, with geometry-group statistics and
   filenames, but must not prevent loading or sequence switching. If the user
   explicitly creates a segment while no segmentation exists, create a generic
   segmentation for the current volume. Record the selected volume, segmentation,
   and segment identities explicitly instead of relying on naming conventions.
   Recording metadata must include all loaded/involved volume sequences with
   sequence index, node ID, and name. Volume switching should be present in raw
   logs and interpreted as compact semantic `volume_change` events.
   Volume node names should be normalized to the full storage filename,
   including suffix such as `.nii.gz`. Re-importing the same scalar volume file
   should replace the older node from the same storage path instead of keeping
   Slicer's `_1` duplicate.
   `A`/`W` and `Z`/`C` wrap around at the ends. Volume switching preserves the
   native Slicer slice view state and should avoid `FitSliceToAll()` in the fast
   path.

6. Current shortcut map.

   - `A` / `W`: next / previous loaded volume sequence.
   - `Z` / `C`: previous / next segment.
   - `V`: show/hide current segment.
   - `Q`: show/hide other segments.
   - `1`: activate Brush tool.
   - `2`: activate Erase tool.
   - `3`: toggle positive prompt point placement.
   - `4`: toggle negative prompt point placement.
   - `E`: toggle SPX boundary visibility.

7. Audio recording is wired to the recording UI via `_AudioSubprocess`.

   The Recording section has three rows:
   - Row 1: `Record` toggle · `M+K` checkbox · `Audio` checkbox
   - Row 2: `Pause` button · `Export` button
   - Row 3: `Audio Input:` label · device dropdown

   Checking only `Audio` enters audio-only mode: no mouse events are recorded,
   and all annotation tools are locked (read-only) for the session. A popup
   confirms before starting. Unlocking happens automatically on stop.

   Checking only `Mouse+Key` prompts whether to also enable audio.

   Audio is captured via `_AudioSubprocess`, which forks `core/_audio_subprocess.py`
   as a separate OS process (CREATE_NO_WINDOW on Windows). The subprocess records
   through `sounddevice.InputStream` at 22050 Hz mono, polls a sentinel stop-file
   every 50 ms, and drains 150 ms of buffered audio after stop. If `sounddevice`
   is unavailable, audio silently skips and mouse recording continues unaffected.

   `_AudioSubprocess.start_time` is set when `start()` is called. This timestamp
   is used on export to trim the prewarm lead-in from the saved WAV (see
   `_finalize_wav` below).

   On export, `_finalize_wav(wav_path)` performs two operations in order:
   1. **Prewarm trim**: discards frames captured before `_recording_start_time`
      by computing `trim_sec = max(0, recording_start - audio.start_time)`.
   2. **Pause silence**: zeros byte ranges for each `(pause_sec, resume_sec)`
      entry in `_pause_intervals`, where both offsets are relative to
      `_recording_start_time` (i.e., the post-trim timeline).

   `_pause_intervals: list[tuple[float, float]]` accumulates across all pause/
   resume cycles in a session. It is cleared at the start of each new recording.

   The saved WAV filename is `{base}_{YYYYMMDDTHHMMSSMMM}.wav`, where the
   timestamp is `_recording_start_time` with millisecond precision. Colons are
   never written to filenames.

   `core/_audio_recorder.py` remains available as standalone chunk-based
   infrastructure for future local Whisper transcription. It is not used by the
   Slicer UI; keep it importable without `sounddevice`.

## Recording Contract

`core/_mouse_recorder.py` records only events that occur in Red/Green/Yellow
slice views and map inside the active volume.

Current schema is intentionally strict while this branch is under test:

- Saving writes both a raw source log (`*_raw.json`) and a compact process log.
  The compact log must be derived from the raw log by `TimeLogInterpreter`.
  A third-stage `TimeLogSummarizer` converts the compact process log into a
  human-readable `annotation_summary` with higher-level activity spans.
  Both `TimeLogInterpreter` and `TimeLogSummarizer` run offline with no Slicer
  dependency.
- The interpreted annotation-process export uses IJK as the compact coordinate
  for cursor-derived events. Mouse entries in the raw-input export intentionally
  omit RAS/IJK and store the original input facts: view, VTK device XY,
  optional global screen XY, pressed state, and wheel delta when relevant.
  Raw source events also include tool changes, segment lifecycle/selection,
  model changes, brush parameter changes, and markups point events.
- `metadata.volume` stores both `ijk_to_ras` and `ras_to_ijk` (its pre-computed
  inverse). Markup point events may carry world RAS because Slicer provides the
  accepted point location that way; their cleaned IJK is derived with
  `ras_to_ijk`.

### Pause/resume during recording

Calling `recorder.pause()` sets `_paused = True`. While paused:
- `_on_mouse` still tracks button state (`_active_mouse_press`) but drops all
  records; it does not append to `recorder.records`.
- `_append` returns immediately without writing.

Calling `recorder.resume()` clears `_paused`. `is_paused` is `True` only when
both `is_active` and `_paused` are `True`.

On the widget side, `_do_pause_recording()` opens a modal `QDialog` that blocks
parent-window input. "Keep Waiting" keeps the dialog open; "Resume" closes it
and calls `_do_resume_recording()`. The dialog is the only mechanism for
resuming — there is no keyboard shortcut or second code path.

`_do_resume_recording()` appends `(pause_sec, resume_sec)` to `_pause_intervals`
(relative to `_recording_start_time`) and calls `recorder.resume()`.

The recorder is an event-listener-style component. It should first produce a
correct in-volume event stream, then annotate events with
annotative/non-annotative classification. Do not let policy decisions drop raw
press/release/move events inside volume views.

Exactly one `metadata` record is created when recording starts. It caches volume
metadata, the 60 Hz sample rate, the initial Red/Green/Yellow visual state
(including each view's DataProbe-style 4x4 `xy_to_ijk` matrix when available),
the `ras_sources` dictionary explaining coordinate sources, and whether
recording began with the left mouse button already down.

Metadata is not an exported event. The compact export stores:

- `metadata.start_time`: absolute time for the recording.

Boundary events (one per stroke boundary or point action):

- `press`: `{id, event, timestamp, ijk, view, tool, segment, brush_mm}`.
  `brush_mm` only present for `brush`/`erase` tools.
- `hold`: `{id, event, timestamp:[…], ijk:[[i,j,k],…], view}`.
  All in-volume annotation samples collapsed into arrays. `segment`, `tool`, and
  `brush_mm` are omitted — they are carried on the preceding `press`.
- `release`: `{id, event, timestamp, ijk, view}`.
- `place`: new point placed. `{id, event, timestamp, ijk, view, segment, point,
  point_name, point_index, negative}`.
- `replace`: point relocated. Same fields as `place`.
- `remove`: point deleted. Same fields as `place`.

Delta events (emitted only when the value changes):

- `slice_change`: `{id, event, timestamp, view, slice}`. Emitted before the
  next boundary event when the active slice in a view shifts.
- `tool_change`: `{id, event, timestamp, view, tool}`. Emitted when the active
  tool changes within a view.
- `brush_change`: `{id, event, timestamp, view, tool, brush_mm}`. Emitted when
  the brush/erase radius changes.

IJK coordinates are derived from VTK device XY via the stored DataProbe-style
`xy_to_ijk` matrix. Point events prefer the pre-computed IJK on the raw event
(derived from world RAS via `metadata.volume.ras_to_ijk`), falling back to
`mouse_boundary.xy` + `xy_to_ijk`. Idle/hover trajectory is omitted entirely.

Movement considers the latest in-volume cursor position at 60 records/sec and
adaptively thins writes using cached XY-to-IJK scale. Pressed annotation moves
target at least 0.5 IJK voxels, clamped to 1-4 px; released hover moves target
at least 2 IJK voxels, clamped to 2-12 px. Time caps keep annotation moves at
least every 100 ms and hover moves at least every 250 ms. The compact policy
constants are stored in `metadata.move_thinning`.

Each slice view owns one recording listener. The listener uses only VTK
interactor `GetEventPosition()` device XY, the same coordinate source consumed
by Slicer DataProbe. Qt event-filter capture is intentionally not used because
it can report a different origin. Avoid heavy work in raw mouse callbacks.
The active-volume hot-path check uses cached per-view DataProbe-style XY-to-IJK
transforms, so inactive XY is dropped before it accumulates, while export still
performs the same IJK-bounds validation. The timer only appends the latest
cached valid sample. Boundary events should flush pending movement before
appending themselves so event order remains meaningful.

Brush/erase press is a required boundary event. If Slicer drops the initial
press but a brush/erase drag sample or release is observed, infer a `press`
boundary before the first sampled move/release and set
`payload.boundary_source`.
Brush/erase movement is classified from the mouse button state: held-button
movement is `annotation_move`; released-button movement is
`non_annotation_move`. Both are listener-recordable in volume views.

Point placement is captured as a raw markups-node source event and interpreted
as one compact `point_placement` boundary, not as raw press plus release mouse
boundaries. A new point records its source verdict on
`PointPositionDefinedEvent`, using the defined control-point location. The
later interaction end is used only to refresh cached metadata and must not
duplicate the verdict. Include point ID and point name/label when available.
Minor drift between press and release is `non_annotation_move` trajectory.

Point relocation is also semantic annotation process data. Record markups
`PointStartInteractionEvent` as `point_drag_start` with `point_action: grab`;
record `PointEndInteractionEvent` as `point_replaced` (distinct event type)
with `point_action: replace` at the final RAS. This separates new placements
(`point_placed`) from relocations (`point_replaced`) at the event-type level,
not just through the `point_action` field. Sample `PointModifiedEvent` movement
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

## Robustness and Predictability

**Rule: No silent failures. No fallback assignments. Every unexpected state is an error.**

This module follows a strict no-silent-failure policy. The goal is that bugs
surface immediately at the point of cause, not later as incorrect behavior.

### Mandatory error on unexpected state

1. **Raise on missing internal attributes.**  
   Every attribute used by a method must be initialized in `__init__`. Methods
   must not patch a missing attribute with a fallback (`if not hasattr(self, …):
   self.x = {}`). If the attribute is absent, the `AttributeError` is the correct
   outcome — it identifies the missing initialization.

2. **Raise on required data that is absent.**  
   When a method depends on a value that must be present (a segment ID in a cached
   point record, a node tag set during initialization), it must raise `RuntimeError`
   if that value is falsy or missing. It must not silently substitute a UI fallback
   such as `currentSegmentID()`. A silent substitute converts a detectable bug into
   invisible wrong behavior.

3. **Do not swallow exceptions.**  
   `except Exception: return` and `except Exception: pass` are forbidden. If a
   statement can raise, let the exception propagate. Slicer prints Python
   tracebacks from event callbacks to the console — that is the correct visibility
   for unexpected errors, not a silent no-op.

4. **No defensive `getattr` fallbacks on own attributes.**  
   `getattr(self, '_foo', default)` and `getattr(widget, '_bar', default)` are
   only acceptable for attributes owned by external Slicer/VTK/Qt objects whose
   existence is version-dependent. For attributes this module controls,
   use direct access. A missing attribute must raise `AttributeError`.

### When `None` / empty returns are legitimate

- A method that queries optional Slicer state (e.g. no volume selected, no
  segment selected) may return `None` or an empty string — these represent valid
  absent-selection states, not errors. The caller must check and decide.
- Early returns inside MRML event callbacks are acceptable when the event fires
  during a known transient state (e.g. `PointPositionDefinedEvent` with no
  defined control point during continuous placement preview). Log a `debug`
  message so the path is visible.

### Rationale

Silent fallbacks turn initialization bugs into data-integrity bugs: the wrong
segment is recorded, the wrong point is tracked, and the error is only visible
long after the code that caused it ran. An immediate `RuntimeError` or
`AttributeError` is always cheaper to debug than a subtly wrong recording.

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
