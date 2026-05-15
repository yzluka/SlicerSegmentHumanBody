# Project Structure

> **Branch:** This document describes the `annotation-process-recorder` branch.
> The architecture here — native Segment Editor wrapping, mouse-centered recording,
> audio capture, and offline analysis pipeline — is specific to this branch and
> differs from `main`.

## Contents

- [What this project is](#what-this-project-is)
- [Two Python environments](#two-python-environments)
  - [Environment A — inside Slicer](#environment-a--inside-slicer-pythonslicerexe)
  - [Environment B — standard Python](#environment-b--standard-python)
  - [Environment C — OS subprocess](#environment-c--os-subprocess-bridge)
- [Dependencies](#dependencies)
- [What is active vs building block](#what-is-active-vs-building-block)
- [Architecture overview](#architecture-overview)
  - [Flow 1 — Annotation](#flow-1--annotation-editing)
  - [Flow 2 — Recording](#flow-2--recording)
  - [Flow 3 — Audio](#flow-3--audio)
  - [Building block — Model framework](#building-block--model-framework)
  - [Offline analysis tool](#offline-analysis-tool-separate-process-no-slicer)
- [Module reference](#module-reference)
  - [SegmentHumanBody.py](#segmenthumanbodypy)
  - [core/_input.py](#core_inputpy--handler-mutual-exclusion)
  - [core/_mouse_recorder.py](#core_mouse_recorderpy--60-hz-event-recorder)
  - [core/TimeLogInterpreter.py](#coretimeloginterpreterpy--raw--process-log-offline)
  - [core/TimeLogSummarizer.py](#coretimelogsummarizerpy--process-log--summary-offline)
  - [core/_audio_subprocess.py](#core_audio_subprocesspy--microphone-capture-subprocess)
  - [core/utils.py](#coreutils-py--coordinate-and-image-utilities-pure-python)
  - [core/modelFamilies.py and modelRegistry.py](#coremodelfamiliespy-and-coremodels--model-framework)
  - [core/models/default.py](#coremodelspy--building-block-identity-model)
  - [core/_logic.py](#core_logicpy--building-block-spx--prompt--custom-undo)
  - [core/_tracker.py](#core_trackerpy--building-block-zero-copy-segment-write-path)
  - [core/_state.py](#core_statepy--building-block-render-loop-gating)
  - [core/_audio_recorder.py](#core_audio_recorderpy--building-block-chunk-based-recorder)
  - [core/_point_log.py](#core_point_logpy--building-block-prompt-point-store)
  - [core/_deps.py](#core_depspy--building-block-lazy-dependency-checker)
  - [core/models/spx.py](#coremodels-spxpy--building-block-superpixel-models)
  - [core/models/timed_annotator.py](#coremodelstimed_annotatorpy--building-block-point-logger)
  - [tools/audio_processor/](#toolsaudio_processor--offline-transcription-tool)
- [Recording data flow](#recording-data-flow)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Other documents in this repository](#other-documents-in-this-repository)

---

## What this project is

SegmentHumanBody is a 3D Slicer extension for medical image annotation research.
It wraps Slicer's native Segment Editor (brush, erase, prompt points, undo/redo)
and adds a second layer on top: a mouse-centered recorder that captures *how*
annotators work, not just the final masks.

The recording pipeline produces three outputs from a single session:

| File | What it contains |
|---|---|
| `*_raw.json` | Every mouse event, tool change, and markup at full resolution |
| `*.json` | Compact semantic log: strokes, clicks, point placements, navigation spans |
| `*_summary.txt` | Human-readable activity timeline |

An optional microphone recording is saved alongside as `*.wav`. A separate
offline tool (`tools/audio_processor/`) can transcribe the WAV and align phrases
to annotation spans.

---

## Two Python environments

This is the most important thing to understand before touching any file.

### Environment A — inside Slicer (`PythonSlicer.exe`)

Slicer ships its own CPython interpreter. It provides three bindings that do not
exist in any standard Python installation:

| Binding | Provides |
|---|---|
| `slicer` | MRML scene, volumes, segmentations, markup nodes, Segment Editor |
| `vtk` | VTK image pipeline, transforms, 4×4 matrices |
| `qt` | PythonQt wrapper around Qt5 widgets |

Everything under `SegmentHumanBody/` runs in this interpreter, including the
test suite:

```powershell
cd D:\SlicerSegmentHumanBody\SegmentHumanBody
PythonSlicer.exe -m pytest tests/ -q
```

Modules that import `slicer`/`vtk`/`qt` **at module level** cannot be used
outside Slicer. Modules that avoid those at module level can be tested or
scripted in plain Python even though they live inside `SegmentHumanBody/`.

### Environment B — standard Python

`tools/audio_processor/` has no Slicer, VTK, or Qt dependency. Install and run
it with a regular Python installation:

```
pip install -r tools/audio_processor/requirements.txt
python -m tools.audio_processor
```

### Environment C — OS subprocess (bridge)

`core/_audio_subprocess.py` is launched as a **separate OS process** by the
Slicer widget. It runs under `PythonSlicer.exe` but has no live Slicer scene
access — it only uses `sounddevice` and `wave`. It must be installed into
Slicer's Python:

```powershell
PythonSlicer.exe -m pip install sounddevice
```

If `sounddevice` is absent the subprocess exits immediately; the widget detects
this and continues without audio.

---

## Dependencies

### Slicer module

| Package | Used in | Install | Required? |
|---|---|---|---|
| `slicer`, `vtk`, `qt` | All UI/widget code | Bundled with 3D Slicer | Yes |
| `sounddevice` | `_audio_subprocess.py` | `PythonSlicer.exe -m pip install sounddevice` | Optional |
| `scikit-image` | `models/spx.py`, `utils.py` | `PythonSlicer.exe -m pip install scikit-image` | Optional (SPX models only) |
| `scipy` | `utils.py` fallback | `PythonSlicer.exe -m pip install scipy` | Optional |
| `pytest` | Test suite | `PythonSlicer.exe -m pip install pytest` | Dev only |

### Offline tool

| Package | Used in | Install | Required? |
|---|---|---|---|
| `faster-whisper>=1.0.0` | `processor.py` | `pip install faster-whisper` (GPU requires CUDA 12) | Yes |
| `tkinter` | `app.py` | Bundled with CPython | Yes |

---

## What is active vs building block

Several modules exist in this branch as foundations for future functionality
but are not wired to the current UI. They are tested and complete — not
abandoned — but a new developer should know which ones actually run during
a session.

| Module | Status | Notes |
|---|---|---|
| `SegmentHumanBody.py` | **Active** | Widget, audio subprocess, widget-facing logic |
| `core/_input.py` | **Active** | Handler mutual exclusion |
| `core/_mouse_recorder.py` | **Active** | 60 Hz recording |
| `core/TimeLogInterpreter.py` | **Active** | Offline, called on export |
| `core/TimeLogSummarizer.py` | **Active** | Offline, called on export |
| `core/utils.py` | **Active** (partially) | `next_segment_name` used directly; SPX helpers are building blocks |
| `core/modelFamilies.py` | **Partially active** | `FAMILY_REGISTRY` imported; only `DefaultFamily` instantiated |
| `core/modelRegistry.py` | **Partially active** | Only `IdentityModel` loaded at runtime |
| `core/models/default.py` | **Active** | `IdentityModel` runs as the no-op template |
| `core/_audio_subprocess.py` | **Active** | Forked by widget as OS process |
| `core/_logic.py` | **Building block** | Not imported by widget; SPX/prompt/undo infrastructure for future model branch |
| `core/_tracker.py` | **Building block** | Zero-copy VTK write path; only used by `_logic.py` |
| `core/_state.py` | **Building block** | Render-loop gating; only referenced in integration test scaffolding |
| `core/_audio_recorder.py` | **Building block** | Chunk-based recorder for future local Whisper integration |
| `core/_point_log.py` | **Building block** | Prompt-point store; not imported by current widget |
| `core/_deps.py` | **Building block** | Lazy dep checker; used by SPX models when activated |
| `core/models/spx.py` | **Building block** | SPX algorithms; registered but UI hidden |
| `core/models/timed_annotator.py` | **Building block** | Point logger with mirror nodes; UI hidden |
| `modelFamilies.SPXModelFamily` | **Building block** | Full SPX expansion logic; buttons hidden |
| `modelFamilies.TimedAnnotatorFamily` | **Building block** | Delegates to `TimedAnnotatorModel`; buttons hidden |
| `modelFamilies.SAMFamily` | **Building block** | Stub placeholder |
| `modelFamilies.AutoModelFamily` | **Building block** | Stub placeholder |

---

## Architecture overview

Three independent flows run inside Slicer during a session. They share no
state and can be understood separately. Only these flows are active in this
branch.

### Flow 1 — Annotation (editing)

The user picks a tool; the widget attaches a handler; the handler activates
the corresponding Slicer Segment Editor effect. Undo/redo delegates entirely
to Slicer's native undo stack — no custom delta tracking in this branch.

```
User clicks tool button
  → Widget
  → core/_input.py          enforce mutual exclusion, ensure segment exists
  → Slicer Segment Editor   Paint / Erase / Markups (native Slicer)
  → Slicer undo stack       Ctrl+Z / Ctrl+Shift+Z handled natively
```

### Flow 2 — Recording

Mouse events are captured at 60 Hz independently of the editing flow.
The recorder writes to its own list and never touches segment buffers.

```
VTK interactor events (Red / Green / Yellow slice views)
  → core/_mouse_recorder.py    XY → in-volume check → adaptive thinning
                               appends to records[] (dropped if paused)

User clicks Export
  → MouseEventRecorder.export()           writes *_raw.json
  → TimeLogInterpreter (offline)          *_raw.json → *.json
  → TimeLogSummarizer  (offline)          *.json     → *_summary.txt
```

Both interpreter and summarizer are pure Python and run with no Slicer
dependency — they read the stored `xy_to_ijk` matrices from the raw log.

### Flow 3 — Audio

Audio runs in a completely separate OS process so a UI freeze cannot drop
microphone frames. The widget side only manages start/stop and WAV finalization.

```
Widget._do_start_recording()
  → _AudioSubprocess.start()    fork core/_audio_subprocess.py
                                (sounddevice → WAV, polls sentinel file)
                                records start_time

User clicks Pause
  → QDialog (modal)             blocks parent window
  → on Resume: append (pause_sec, resume_sec) to _pause_intervals

User clicks Export
  → _AudioSubprocess.stop()     write sentinel file, wait for process exit
  → _finalize_wav()
        trim  = recording_start − audio_subprocess.start_time  (prewarm)
        zero  = bytes in each _pause_intervals entry           (pauses)
        write trimmed + silenced WAV
```

### Building block — Model framework

The model family / registry infrastructure exists for a future branch that
will add inference-based segmentation. In this branch only `DefaultFamily` +
`IdentityModel` are instantiated on startup as a no-op placeholder. All other
families have their UI buttons hidden.

When a future branch activates a family, no widget code needs changing —
only the family's `VISIBLE_BUTTONS` set determines which buttons appear.

```
core/modelFamilies.py    FAMILY_REGISTRY, VISIBLE_BUTTONS
  → core/modelRegistry.py    lazy singleton cache
  → core/models/
        default.py            IdentityModel (active no-op)
        spx.py                SLIC / Felzenszwalb / Tester  [building block]
        timed_annotator.py    point logger with mirror nodes [building block]
```

### Offline analysis tool (separate process, no Slicer)

```
tools/audio_processor/app.py       tkinter GUI — two-stage workflow
  Stage 1 · Transcribe
  → processor.transcribe_and_phrase()
        load annotation JSON        (auto-interprets raw / process / summary)
        transcribe WAV              faster-whisper → whisper_{stem}.json
        merge words → phrases       silence gaps + annotation boundaries
        write phrases_{stem}.txt    one phrase per line, editable

  Stage 2 · Review Phrases & Generate Reports
        user edits phrases_{stem}.txt in any text editor
  → processor.apply_phrase_corrections()
        edit-distance alignment     SequenceMatcher aligns corrected tokens to
                                    original words; unchanged tokens keep exact
                                    timing + probability; changed tokens get
                                    synthetic entries with proportionally
                                    distributed timestamps
        write whisper_{stem}_refined.json
  → processor.process()
        re-merge with refined words
        clean via cleaner.py        regex replacements + review flags
        align phrases to spans      timestamp overlap
        write _transcript.json      structured output
            _transcript.txt         audio-centric view
            _caption.txt            annotation-centric view
```

---

## Module reference

### `SegmentHumanBody.py`

The Slicer module entry point. Three responsibilities:

**`_AudioSubprocess`** — manages the per-session microphone subprocess. Starts a
prewarm process so the mic is ready before Record is clicked. Tracks
`start_time` (when `start()` was called) so `_finalize_wav` can trim prewarm
audio. On export: trims frames before `_recording_start_time`, then silences
every `(pause_sec, resume_sec)` interval in `_pause_intervals`.

**`SegmentHumanBodyWidget`** — all Qt signal wiring, volume/segment selectors,
handler lifecycle, recording/pause/export buttons, keyboard shortcuts. Does not
implement any editing logic itself; delegates to `_input` and `_mouse_recorder`.

**`SegmentHumanBodyLogic`** — `ScriptedLoadableModuleLogic` subclass defined
at the bottom of the same file. Handles volume/segmentation lifecycle, scene
navigation, geometry validation, Segment Editor node wiring, and prompt node
management. This is the active logic class — it is separate from
`core/_logic.py`, which is a building block and not imported by the widget.

---

### `core/_logic.py` — building block: SPX / prompt / custom undo

**Not imported by the current widget.** Contains the infrastructure for a
future branch that adds inference-based segmentation:

- `SegmentTracker` cache — replaces when the active segment or volume changes.
- Custom stroke delta pipeline — `capture_current_slice()`, `commit_stroke()`,
  `reverse_change()` / `forward_change()` for a custom undo stack independent
  of Slicer's native one.
- SPX expansion — calls `family.on_expand()`, feeds label maps to
  `select_spx_labels()`, writes results via `SegmentTracker`.
- Coordinate conversion and prompt markup node management.

Imports: `slicer`, `vtk`, numpy, `_tracker`, `modelFamilies`, `utils`.

---

### `core/_tracker.py` — building block: zero-copy segment write path

**Not used in the current UI.** Only imported by `core/_logic.py`.

Designed to be the single write path for segment mask modifications when
custom inference is active. Holds a direct numpy view into the VTK image
buffer so writes are in-place. Stores sparse `MaskChange` deltas (int16
bounding-box crops) for a custom undo stack. Falls back to the MRML pipeline
when the direct VTK path is unavailable.

Imports: `slicer`, `vtk.util.numpy_support`, numpy, `utils`.

---

### `core/_state.py` — building block: render-loop gating

**Not used in the current UI.** Only referenced in integration test scaffolding
(`Testing/Python/SegmentHumanBodyTest.py`).

`WidgetState` provides nestable pause/resume and re-entrant render dispatch
(queues one pending pass rather than dropping or re-entering). Designed for
a branch that runs custom rendering passes triggered by mouse events.

---

### `core/_input.py` — handler mutual exclusion

`BrushHandler`, `EraseHandler`, and `PointHandler` each wrap one Slicer tool.
`attach()` enforces: volume check → detach previous → ensure segment exists →
activate Slicer tool. `detach()` deactivates the wrapped Slicer tool.

Only one handler is active at a time. Switching tools calls `detach` on the
outgoing handler before `attach` on the incoming one.

`PointHandler` requires `widget._active_prompt_widget` to be set before
`attach()` — raises `RuntimeError` immediately if it is `None`.

Imports: `slicer`.

---

### `core/_mouse_recorder.py` — 60 Hz event recorder

Runs inside Slicer. Captures raw device events from Red/Green/Yellow slice views.

- Each slice view gets one VTK interactor listener using `GetEventPosition()`
  device XY — the same coordinate source as Slicer DataProbe.
- A 60 Hz timer applies adaptive XY-to-IJK thinning: annotation moves (button
  held) target ≥0.5 IJK voxels; hover moves target ≥2 IJK voxels.
- Records: mouse trajectory, button events, tool changes, brush parameters,
  segment lifecycle, point placements/relocations/removals, volume changes,
  scroll events.
- Pause/resume: `_paused` flag drops records in `_on_mouse` and `_append` while
  still tracking button state.
- `export()` writes `*_raw.json` containing all records plus per-view
  `xy_to_ijk` matrices needed for offline IJK derivation.

Imports: `slicer`, `vtk`, `qt`.

---

### `core/TimeLogInterpreter.py` — raw → process log (offline)

No Slicer dependency. Reads `*_raw.json`, applies the stored `xy_to_ijk`
matrices to derive IJK, classifies events as boundary or trajectory, and emits
the compact `annotation_process` JSON. Runs on any machine.

---

### `core/TimeLogSummarizer.py` — process log → summary (offline)

No Slicer dependency. Reads `annotation_process` JSON, groups consecutive
events into named activity spans (`stroke`, `click`, `point_click_place`,
`point_drag`, `volume_navigation`, `slice_navigation`), and produces a
human-readable `*_summary.txt`. Carries forward running state (volume, tool,
segment, view, slice, brush_mm) across spans.

Each span stores `start_id` and `end_id` — the event ID range from the compact
log that it covers. These are the cross-reference handles for future
expand-on-demand detail.

---

### `core/_audio_subprocess.py` — microphone capture subprocess

Launched as a separate OS process by `_AudioSubprocess` in the widget. Records
via `sounddevice.InputStream` at 22050 Hz mono. Polls a sentinel stop-file every
50 ms. After stop, drains 150 ms of buffered audio, writes the WAV, and exits
with a result JSON. No Slicer scene access.

---

### `core/_audio_recorder.py` — building block: chunk-based recorder

**Not used in the current UI.** The active UI uses `_AudioSubprocess` instead.

Pure Python. Designed for a future local Whisper workflow: splits microphone
audio into 30-second timestamped WAV chunks, writes a JSON manifest, and
provides `merge_chunks_to_wav()` to concatenate chunks into a single file.
Lazy-imports `sounddevice` only in `start()` so the module is safe to import
without audio hardware.

---

### `core/_point_log.py` — building block: prompt-point store

**Not imported by the current widget.**

Pure Python. Maps segment IDs to ordered lists of `{ras, is_neg, cp_id}`
entries. Designed to persist prompt-point state across segment switches when
the prompt-point / SPX workflow is active. `sync_removed()` reconciles stale
entries per polarity per `PointRemovedEvent`. `export()` returns a deep copy
safe for JSON serialisation.

---

### `core/_deps.py` — building block: lazy dependency checker

**Not used in the current UI.** Active only when SPX models are loaded.

Process-scoped cache that probes each package once and caches the result.
Lets SPX models defer `scikit-image` import until actually needed rather than
failing at module load time.

---

### `core/utils.py` — coordinate and image utilities (pure Python)

No Slicer/VTK imports. The widget currently uses only `next_segment_name`
directly. The rest of the module is infrastructure consumed by the building-block
modules (`_logic.py`, `_tracker.py`, SPX models) when they are active.

- **Active now** — `next_segment_name` (segment naming).
- **Building block** — `ras_to_ijk_3d`, `ras_to_ijk_2d` (coordinate conversion);
  `get_slice_from_volume` / `write_slice_to_volume` (zero-copy numpy views);
  `apply_window_level` (uint8 normalization); SPX helpers (`select_spx_labels`,
  `labels_at_points`, `spx_boundary_mask`, `extract_connected_component`);
  `parse_user_parameters`, `call_if_exists`.

---

### `core/modelFamilies.py` and `core/modelRegistry.py` — model framework

`ModelRegistry` is a session-scoped singleton that lazy-instantiates model
objects on first access and caches them. Adding a model requires one entry in
`_MODEL_FACTORIES`.

`modelFamilies.py` defines family classes, each controlling:
- `VARIANTS` — which model variants are selectable.
- `VISIBLE_BUTTONS` — which UI widget names are shown when this family is
  active. The widget reads this set and shows/hides buttons accordingly — no
  widget code needs changing when a new family is added.
- Hooks (`onRender`, `on_expand`, `on_assign_2d`, etc.) called by the widget at
  the appropriate moments.

`FAMILY_REGISTRY` at the bottom of `modelFamilies.py` is the single place to
register a new family.

| Family | Status | Models |
|---|---|---|
| `DefaultFamily` | **Active** — instantiated on startup | `IdentityModel` — returns input unchanged |
| `SPXModelFamily` | Building block — UI buttons hidden | SLIC, Felzenszwalb, Tester |
| `TimedAnnotatorFamily` | Building block — UI buttons hidden | `TimedAnnotatorModel` |
| `SAMFamily` | Building block — stub placeholder | — |
| `AutoModelFamily` | Building block — stub placeholder | — |

---

### `core/models/default.py` — identity model

No-op template. `forward(**kwargs)` returns `kwargs['img']` unchanged. Reference
implementation for the model contract: must have `PARAM_HINT`, `DOC_URL`, and
`forward()`.

---

### `core/models/spx.py` — building block: superpixel models

**UI hidden in this branch.** Registered in `ModelRegistry` but not loaded.

Three `SPXModel` implementations, each taking a 2-D numpy image and returning
an integer label map: `SPX_Tester2D` (uniform grid, no deps), `SPX_SLIC2D`
(scikit-image SLIC), `SPX_Felzenszwalb2D` (graph-based; handles
constant-intensity edge case). scikit-image is checked at `__init__()` via
`_deps.py`, not at import time.

---

### `core/models/timed_annotator.py` — building block: point logger

**UI hidden in this branch.** Registered in `ModelRegistry` but not loaded.

Session-persistent model that records timestamped annotation points and
mirrors them as `vtkMRMLMarkupsFiducialNode`s in the Slicer scene. Tracks
position history per point across drags. Exports nested JSON with both RAS
and IJK coordinates. Used by `TimedAnnotatorFamily`.

Imports `slicer` / `vtk` / `qt` inside methods (not at module level).

---

### `tools/audio_processor/` — offline transcription tool

Standalone, no Slicer. Requires `faster-whisper`.

**`processor.py`** — full pipeline:
1. Load annotation JSON (auto-interprets `raw` or `process` inputs via
   `TimeLogInterpreter` + `TimeLogSummarizer` if needed).
2. Transcribe WAV with Whisper at word granularity.
3. Merge words into phrases using silence gaps and annotation event boundaries.
4. Align phrases to annotation spans by timestamp overlap.
5. Write `_caption.txt` (annotation-centric), `_transcript.txt`
   (audio-centric), `_transcript.json`.

**`cleaner.py`** — post-processing layer. Applies case-insensitive regex
corrections from JSON files in a `corrections/` directory to transcribed text.
Flags phrases matching `_review_patterns.json` for manual review.

**`app.py`** — tkinter GUI. File pickers for JSON and WAV; auto-matches WAV by
parsing the timestamp embedded in the filename (`_YYYYMMDDTHHMMSSmmm.wav`).
Runs the pipeline in a background thread.

---

## Recording data flow

```
1. Session start
   Widget._do_start_recording()
     → MouseEventRecorder.start()     — installs VTK listeners on slice views
     → _AudioSubprocess.start()       — forks audio subprocess (prewarm)

2. During annotation
   VTK interactor events → _on_mouse()
     → _active_region_gate check (is XY inside active volume?)
     → if _paused: track button state only, drop record
     → else: append to records[]

3. Pause
   _do_pause_recording()
     → recorder.pause()               — sets _paused = True
     → QDialog.exec_()                — modal blocks parent window
     — user clicks Resume →
   _do_resume_recording()
     → appends (pause_sec, resume_sec) to _pause_intervals
     → recorder.resume()              — clears _paused

4. Export
   _save_recording_to_user_path()
     → MouseEventRecorder.export()    — writes *_raw.json
     → TimeLogInterpreter             — reads *_raw.json, writes *.json
     → TimeLogSummarizer              — reads *.json, writes *_summary.txt
     → _AudioSubprocess.stop()        — signals subprocess via sentinel file
     → _finalize_wav()
         trim_sec = recording_start - audio_subprocess.start_time
         frames = frames[trim_bytes:]           (remove prewarm)
         for (s, e) in _pause_intervals:        (silence pauses)
             frames[s:e] = b'\x00' * (e - s)
         write trimmed/silenced WAV
```

---

## Tests

All tests run under Slicer's Python. Tests that stub Slicer calls work without a
live GUI; tests requiring active slice views are skipped automatically.

```powershell
cd D:\SlicerSegmentHumanBody\SegmentHumanBody
PythonSlicer.exe -m pytest tests/ -q
```

Slicer-native integration tests (live slice views, markup placement):

```powershell
& 'C:\Users\82755\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe' --no-main-window --python-script D:/SlicerSegmentHumanBody/run_slicer_tests.py
```

| Test file | What it covers | Needs live Slicer? |
|---|---|---|
| `test_mouse_recorder.py` | `_mouse_recorder.py`, `TimeLogInterpreter` (via export) | Partial (monkeypatched) |
| `test_time_log_summarizer.py` | `TimeLogSummarizer.py` | No |
| `test_audio_recorder.py` | `_audio_recorder.py` | No |
| `test_deps.py` | `_deps.py` | No |
| `test_families.py` | `modelFamilies.py` | No |
| `test_registry.py` | `modelRegistry.py` | No |
| `test_utils.py` | `utils.py` | No |
| `test_spx_models.py` | `models/spx.py` | No |
| `test_segment_lifecycle_recording.py` | Widget segment lifecycle, recording restart | Partial (stubbed) |
| `test_navigation_shortcuts.py` | `A`/`W`/`Z`/`C` navigation logic | Partial (stubbed) |
| `test_pause_recording.py` | Recorder pause/resume, `_finalize_wav`, UI button state | Partial (stubbed) |
| `test_undo_widget.py` | Widget undo/redo flows | Partial (stubbed) |
| `test_point_log.py` | `_point_log.py` | No |
| `test_cleaner.py` | `tools/audio_processor/cleaner.py` | No |

---

## Repository layout

```text
SlicerSegmentHumanBody/
├── README.md
├── PROJECT_STRUCTURE.md
├── LICENSE.md
├── CMakeLists.txt
├── run_slicer_tests.py
├── SegmentHumanBody/
│   ├── SegmentHumanBody.py              — widget + audio subprocess manager
│   ├── CMakeLists.txt
│   ├── CLAUDE.md                        — AI assistant / maintainer guidance
│   ├── SPEC.md                          — behavior specification
│   ├── FRAGILE.md                       — known fragile spots + planned work
│   ├── Resources/UI/SegmentHumanBody.ui
│   ├── core/
│   │   ├── _input.py                    — handler wrappers                   [active]
│   │   ├── _mouse_recorder.py           — 60 Hz event recorder               [active]
│   │   ├── _audio_subprocess.py         — microphone OS subprocess           [active]
│   │   ├── TimeLogInterpreter.py        — raw → process log (offline)        [active, pure python]
│   │   ├── TimeLogSummarizer.py         — process log → summary (offline)    [active, pure python]
│   │   ├── utils.py                     — coord/image/spx utilities          [partial: next_segment_name active]
│   │   ├── modelFamilies.py             — family classes + FAMILY_REGISTRY   [partial: DefaultFamily active]
│   │   ├── modelRegistry.py             — lazy model cache                   [partial: IdentityModel active]
│   │   ├── models/
│   │   │   ├── default.py               — IdentityModel                      [active, pure python]
│   │   │   ├── spx.py                   — superpixel algorithms              [building block, needs scikit-image]
│   │   │   └── timed_annotator.py       — point logger with mirror nodes     [building block, needs slicer]
│   │   ├── _logic.py                    — SPX/prompt/undo orchestrator       [building block, needs slicer]
│   │   ├── _tracker.py                  — zero-copy VTK write path           [building block, needs slicer]
│   │   ├── _state.py                    — render-loop gating                 [building block, needs qt]
│   │   ├── _audio_recorder.py           — chunk-based recorder               [building block, pure python]
│   │   ├── _point_log.py                — prompt-point store                 [building block, pure python]
│   │   └── _deps.py                     — lazy dep checker                   [building block, pure python]
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_mouse_recorder.py
│   │   ├── test_time_log_summarizer.py
│   │   ├── test_audio_recorder.py
│   │   ├── test_deps.py
│   │   ├── test_families.py
│   │   ├── test_registry.py
│   │   ├── test_utils.py
│   │   ├── test_spx_models.py
│   │   ├── test_segment_lifecycle_recording.py
│   │   ├── test_navigation_shortcuts.py
│   │   ├── test_pause_recording.py
│   │   ├── test_undo_widget.py
│   │   ├── test_point_log.py
│   │   └── test_cleaner.py
│   └── Testing/Python/SegmentHumanBodyTest.py   — Slicer-native integration tests
└── tools/
    └── audio_processor/               — standalone, no Slicer
        ├── __main__.py
        ├── app.py                     — tkinter GUI
        ├── processor.py               — Whisper pipeline + alignment
        ├── cleaner.py                 — text correction layer
        ├── requirements.txt           — faster-whisper>=1.0.0
        └── corrections/default/
            ├── anatomy.json
            ├── annotation_actions.json
            └── _review_patterns.json
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `A` / `W` | Next / previous loaded volume sequence (wraps) |
| `Z` / `C` | Next / previous segment in the active segmentation (wraps) |
| `Q` | Show / hide current segment |
| `S` | Show / hide all other saved segments |
| `E` | Reserved — no binding |
| `Ctrl+Z` | Undo (delegates to Slicer's native Segment Editor undo stack) |
| `Ctrl+Shift+Z` | Redo (delegates to Slicer's native Segment Editor redo stack) |

---

## Other documents in this repository

| Document | Location | Purpose |
|---|---|---|
| `README.md` | repo root | User-facing overview: what the extension does, installation, basic usage |
| `PROJECT_STRUCTURE.md` | repo root | This file — developer orientation, architecture, active vs building-block code |
| `CLAUDE.md` | `SegmentHumanBody/` | Guidance for AI assistants and maintainers: development principles, recording contract, robustness rules, audio architecture |
| `SPEC.md` | `SegmentHumanBody/` | Behavior specification: expected outcomes for every user action, recording requirements, coordinate system |
| `FRAGILE.md` | `SegmentHumanBody/` | Known fragile spots, pre-flight checklist before touching sensitive areas, and planned improvements |
```
