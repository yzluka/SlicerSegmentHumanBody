# SPEC.md - Behavior Specification

This file defines expected behavior for the current native-editor-wrapper
branch. Re-check relevant sections after changes to `SegmentHumanBody.py`,
`core/_input.py` or `core/_mouse_recorder.py`.

## 0. Handler Wrappers

All interactive tools are represented by handler wrappers in `core/_input.py`.

| Rule | Expected behavior |
|---|---|
| Mutual exclusion | Attaching one handler detaches the previous handler |
| Wrapper ownership | Detaching a handler disables the Slicer tool it wraps |
| Segment guard | Brush, erase, and point attach all ensure a valid segment exists |
| Prompt nodes | The active segment always has one positive and one negative prompt node |
| Point placement | `PointHandler.attach()` enables markups place mode; `detach()` disables it |

Handler mapping:

| Handler | Wrapped tool |
|---|---|
| `BrushHandler` | Segment Editor `Paint` |
| `EraseHandler` | Segment Editor `Erase` |
| `PointHandler` | `qSlicerSimpleMarkupsWidget` place mode |

## 1. Volume And Segmentation

| # | Action | Expected result |
|---|---|---|
| 1.1 | Select a volume with no segmentation | Segmentation can be created on first segment/tool action |
| 1.2 | Select an existing segmentation | Display node is created if missing; selector is wired and Segment Editor follows the selected volume/segmentation |
| 1.3 | Clear volume selector | Parameter node updates without crash |
| 1.4 | Switch volume | Window/level controls sync from the new volume and Red/Green/Yellow views show the selected volume |
| 1.5 | Switch volume with an existing segmentation selected | Segmentation selection remains manual and unchanged |
| 1.6 | Switch volume with mismatched selected segmentation grid shape, spacing, or orientation | User is asked to create a new empty segmentation, keep the current segmentation, or cancel the switch |
| 1.7 | Import compatible CT-derived volumes where derived volumes have zero origin | The first compatible non-zero origin is copied onto zero-origin volumes |
| 1.8 | Import volumes with inconsistent shape, spacing, or orientation | After import settles, user sees one informational warning with statistics and filenames for all geometry groups; loading and switching remain allowed |
| 1.9 | Start recording with multiple loaded volumes | Metadata stores all loaded sequence indexes, node IDs, and names |
| 1.10 | Switch volume while recording | Raw log stores `volume_changed`; semantic log stores compact `volume_change` |
| 1.11 | Click `Clear Loaded Volumes` | All scalar volume nodes are removed; segmentation nodes remain for manual deletion |
| 1.12 | Re-import the same scalar volume file | New node replaces the older node from the same path and uses the full filename with suffix rather than `_1` |
| 1.13 | Create a segment when no segmentation exists | A generic segmentation is created for the current volume |
| 1.14 | Record segment selection | The raw process log stores explicit volume, segmentation, and segment identities |

## 2. Segment Management

| # | Action | Expected result |
|---|---|---|
| 2.1 | Press `+` | New segment is created and selected |
| 2.2 | Press `A` | Select next loaded scalar volume sequence; does not create a segment or change segmentation selection |
| 2.3 | Activate Brush/Erase/Point with no segment | Segment is created before the wrapped tool activates |
| 2.4 | Press `-` on middle/later segment | Removed; selector switches to the immediate previous segment |
| 2.5 | Press `-` on first segment | Removed; selector switches to the next segment if one exists |
| 2.6 | Press `-` on only segment | Removed; selector is empty |
| 2.7 | Remove segment | Its positive/negative prompt nodes are removed |
| 2.8 | Press `W` | Select previous loaded scalar volume sequence, with wraparound |
| 2.9 | Press `A` / `W` after moving/panning a slice | Selected sequence changes using normalized origins and native Slicer slice state |
| 2.10 | Press `Z` / `C` | Select previous / next segment in the active segmentation, with wraparound |

## 3. Prompt Points

Each segment owns one positive and one negative `vtkMRMLMarkupsFiducialNode`.
Their node IDs are stored as segment tags:

- `SegmentHumanBody.posNodeID`
- `SegmentHumanBody.negNodeID`

| # | Action | Expected result |
|---|---|---|
| 3.1 | Switch segment | Prompt widgets are rewired to that segment's nodes |
| 3.2 | Wire prompt nodes programmatically | Does not accidentally enter point-add mode |
| 3.3 | Switch Brush -> Point | Brush effect is deactivated; point placement is enabled |
| 3.4 | Switch Point -> Brush/Erase | Point placement is disabled; Paint/Erase activates |
| 3.5 | Place point while recording | A semantic `point_placed` verdict event is recorded when Slicer defines the control point |

## 4. Brush And Erase

| # | Action | Expected result |
|---|---|---|
| 4.1 | Activate Brush | Segment Editor Paint effect activates |
| 4.2 | Activate Erase | Segment Editor Erase effect activates |
| 4.3 | Toggle Brush off | Paint effect deactivates; options frame is hidden/returned |
| 4.4 | Toggle Erase off | Erase effect deactivates; options frame is hidden/returned |
| 4.5 | Ctrl+Z | Delegates to Segment Editor undo |
| 4.6 | Ctrl+Shift+Z | Delegates to Segment Editor redo |

## 5. Recording

Recording is mouse-centered. It is not a general UI macro recorder.

| # | Requirement | Expected behavior |
|---|---|---|
| 5.1a | Event ID | Every internal record has a sequential integer `event_id`; compact export writes it as `id` |
| 5.1 | Capture scope | Only Red/Green/Yellow slice-view events are considered |
| 5.1b | Native editor capture | Each slice view owns one recording listener. The listener uses only the VTK interactor `GetEventPosition()` device-XY stream, matching Slicer DataProbe. Qt event-filter capture is intentionally not used because it can produce a different coordinate origin |
| 5.2 | Active volume boundary | Mouse events outside the active volume are ignored using cached per-view DataProbe-style `xy_to_ijk` bounds checks (`sliceView.convertDeviceToXYZ()` plus background-layer `GetXYToIJKTransform()`) |
| 5.3 | Return inside volume | Recording resumes when the mouse maps back inside the volume |
| 5.4 | Movement sampling | In-volume movement is sampled at 60 Hz, then thinned by cached XY-to-IJK scale. Annotation moves target >=0.5 IJK voxels clamped to 1-4 px; hover moves target >=2 IJK voxels clamped to 2-12 px. Time caps keep annotation moves at least every 100 ms and hover moves at least every 250 ms. Policy constants are stored in `metadata.move_thinning` |
| 5.5 | Scroll/wheel | Wheel events record `view_changed` visual trajectory events |
| 5.6 | Mouse status | Records include `move`, `press`, `release`, or `view` |
| 5.7 | Handler state | Raw mouse records carry handler/tool context as deltas; compact export includes only parameters needed to reconstruct the semantic event, such as brush diameter |
| 5.8 | Visualization state | Initial Red/Green/Yellow state is stored once in metadata; later snapshots are stored on infrequent boundary/view events so raw XY can be interpreted offline |
| 5.9 | Points | Point markups source events are kept in `_raw.json`; new point placement becomes compact `place`, relocation drag-end becomes compact `replace`, and removal becomes compact `remove` during offline interpretation |
| 5.10 | Hotkeys/actions | Only segmentation-changing actions should be semantic recording actions |
| 5.11 | Segment creation | Segment creation is recorded as a raw source event, not as an annotation trajectory |
| 5.12 | Segment removal/rename | Active-segmentation `SegmentRemoved` records raw `segment_removed` with last known segment name; name changes record raw `segment_renamed` with old/new names |
| 5.13 | Trajectory classification | Trajectory events include `trajectory_kind` (`annotation_move`, `non_annotation_move`, or `view_change`) and `trajectory_role` (`annotation_trajectory` or `visualization_trajectory`) |
| 5.14 | Point relocation | Existing point drag start records `point_drag_start` with `point_action: grab`; sampled drag movement records `point_drag_move`; release records `point_replaced` with `point_action: replace` at the final RAS; move/end are ignored unless a valid defined-point drag start was accepted; just-defined placement points are not treated as relocation |
| 5.15 | Metadata | Exactly one `metadata` record is created when recording starts; it stores volume metadata, sample rate, initial Red/Green/Yellow visual state, and whether recording started while the left button was already down |
| 5.16 | Live count | Record status count updates as events are appended |
| 5.17 | Segment switching | Segment selection changes record raw `segment_selected`; mouse/tool events still carry active segment context as needed |
| 5.18 | Brush/erase press fallback | If Slicer drops the initial press but a brush/erase drag sample or release is observed, a `press` boundary is inferred before that first sampled move/release |
| 5.19 | Tool selection | Activating a tool records `tool_selected` with `tool: 'brush'`, `'erase'`, or `'point'` and the active `segment_id`. Explicitly deactivating a tool (toggle off without switching to another) records `tool_selected` with `tool: null`. Switching directly from one tool to another records only the activation event for the incoming tool. |
| 5.20 | Recording UI | Row 1: single `Start/Stop Recording` toggle + `Mouse+Key` checkbox + `Audio` checkbox. Row 2: `Audio Device:` label + device dropdown + `Export` button. |
| 5.21 | Recording modes | Both checked: mouse events + audio. Only `Mouse+Key`: prompt asks whether to also enable audio. Only `Audio`: audio-only mode (popup confirmation; annotation tools locked for the session; unlock on stop). Neither checked: warning, no recording started. |
| 5.22 | Audio-only mode | No mouse recorder starts. Brush, erase, add/remove segment, and prompt-point widgets are disabled. Only view navigation and segment/volume switching remain available. `_lock_annotation_tools` is called on start and reversed on stop. |
| 5.23 | Audio capture | `_AudioSubprocess` forks `core/_audio_subprocess.py` (CREATE_NO_WINDOW on Windows) at 22050 Hz mono. Stop is signalled via sentinel file; process drains 150 ms of buffered audio after stop. Missing `sounddevice` silently disables audio; mouse recording continues. |
| 5.24 | WAV export filename | WAV is saved as `{base}_{YYYYMMDDTHHMMSSMMM}.wav` alongside the JSON. The timestamp is the recording start time with millisecond precision; no colons are written to filenames. In audio-only mode, the user's chosen base name gets the same timestamp postfix. |
| 5.25 | Stop vs export | Clicking `Stop Recording` finalises the WAV (stops the subprocess) but keeps temp files. `Export` can be used at any time — it stops recorders if still active, then opens save dialogs. Starting a new recording while unsaved data exists triggers Save/Discard/Cancel. |

Movement capture is staged before policy classification. Each raw move keeps the
original VTK device XY and is kept only if cached DataProbe-style XY-to-IJK
mapping places it inside the active volume; the timer then appends the latest
valid in-volume movement sample at the configured rate.
If the cursor leaves the volume before the timer fires, the last valid
in-volume sample is still eligible to record. Boundary events flush any pending
movement first, so event order follows the user's interaction order.

The core event listener should not drop press/release/move events that occur
inside volume views. Annotative/non-annotative labels are payload
classification, and downstream ignore/export policy should be applied
after the listener has a correct event stream.
For brush/erase specifically, `press` and `release` are boundary events and
always record. `move` while the left button is pressed/held records as
`annotation_move`. `move` with the button released records as
`non_annotation_move`.
For new point placement, the placement location is taken from the defined
markups control point and stored as a verdict `point_placed` event with
`analysis_event_type: boundary_event`, `mouse_status: release`,
`point_action: place`, point ID, and point name/label when available. Minor
movement between press and release is `non_annotation_move` trajectory. For
relocating an already placed point, start/end are `point_drag_start` /
`point_replaced` boundaries with `point_action: grab` / `replace`, and sampled
movement is `non_annotation_move`. Point deletion records a `point_removed`
boundary with the last cached RAS, point ID, point index, segment ID, and
positive/negative point role when available.
The compact export suppresses raw point-tool mouse press/move/release companion
events during an accepted point relocation, because the markups point stream is
the authoritative annotation coordinate stream for that interaction.
Point-drag move throttling occurs before node lookup/RAS extraction. Segment
modified handling only syncs prompt names when the segment name actually
changes. Parameter-node-to-UI sync is guarded against selector feedback loops.

Saving writes two logs. The raw log is the source of truth; the compact process
log is generated from it by `core.TimeLogInterpreter.TimeLogInterpreter`.
`TimeLogInterpreter` must be able to run outside Slicer using only `_raw.json`.

The compact process log is:

- top level: `type`, `metadata`, `events`
- `metadata`: one session-start block with volume metadata (including `ijk_to_ras`
  and `ras_to_ijk`), sample rate, coordinate system (`IJK`), initial
  Red/Green/Yellow visual state, and absolute `start_time`
- each interpreted event: `id`, absolute `timestamp`, `event`, optional `ijk`
  (voxel indices derived from DataProbe-style device XY for cursor events or
  from markup RAS for point events), and only the fields needed for that event

Metadata is not an event. Exported event IDs start at 1 and count only
process events in `events`. Visual-state snapshots intentionally omit repeated
slice-view dimensions; they keep only the state needed to restore view position
and scale.

Mouse movement and boundaries use compact event fields:

- `view`: Red/Green/Yellow slice view
- `mouse`: `move`, `press`, `release`, or `view`
- `pressed`: `1` when the left mouse button is down, `0` when released
- `analysis`: `boundary_event` or `trajectory_event`
- `kind`: `annotation_move`, `non_annotation_move`, or `view_change`
- `role`: `annotation_trajectory` or `visualization_trajectory`
- `tool`: `brush`, `erase`, `point`, or `None`
- `segment`: active segment ID when available
- `brush_mm`: brush/erase radius when relevant
- point fields (`point`, `point_index`, `point_action`, `negative`) when
  relevant
- view fields (`view_event`, `wheel_delta`, `visual_state`) only for
  `view_changed` events

IJK is the canonical compact coordinate for interpreted events.
The separate raw-input export stores original input facts: absolute timestamp,
event, view, VTK device XY, optional global screen XY, pressed state, and wheel
delta when relevant. It also stores source events needed for offline
interpretation: tool selection, segment creation/removal/rename/selection,
model selection, brush parameter changes, and markups point source events.
Mouse raw events intentionally omit RAS/IJK. Markups point source events may
carry `markup_ras` because Slicer provides the accepted point in world
coordinates rather than as a mouse XY event.

Export-time interpretation uses stored per-view `xy_to_ijk` matrices captured
from the same route as Slicer DataProbe. Point-event IJK is derived from markup
RAS with `metadata.volume.ras_to_ijk`.

### Coordinate System

Mouse coordinates are VTK device XY in the slice view and are documented in
`metadata.ras_sources`:

- `cursor` - VTK interactor `GetEventPosition()` device XY interpreted through
  `sliceView.convertDeviceToXYZ()` and the background layer
  `GetXYToIJKTransform()`, matching Slicer DataProbe. Used for mouse `move`,
  `press`, `release`, and `view_changed` events. The 4x4 `xy_to_ijk` matrix for
  each slice view is stored in `metadata.initial_visual_state` and updated in
  `view_changed` events.
- `markup_world` - `vtkMRMLMarkupsFiducialNode.GetNthControlPointPositionWorld()`;
  the actual 3D placement position after Slicer crosshair/picking. Used for
  `point_placed`, `point_replaced`, `point_removed`, `point_drag_start`, and
  `point_drag_move` events.

## 6. Model Family Template

The current active template is:

| Family | Variant | Model | Expected behavior |
|---|---|---|---|
| `Default` | `Identity` | `IdentityModel` | Returns `img` unchanged |

Future families should follow this registry pattern:

1. Add model class/factory.
2. Register it in `modelRegistry.py`.
3. Add a family/variant mapping in `modelFamilies.py`.
4. Wire UI visibility through `VISIBLE_BUTTONS` only when the UI is restored.

## 7. Visibility And Window/Level

| # | Action | Expected result |
|---|---|---|
| 8.1 | Toggle current segment visibility (`Q`) | Only current segment triplet is affected |
| 8.2 | Toggle saved/other segments (`S`) | All non-current segments are affected |
| 8.3 | Switch segment | Incoming segment becomes visible |
| 8.4 | Change W/L controls | Volume display node updates immediately |
| 8.5 | Apply W/L | Same W/L values are explicitly written |

Superpixel-grid display and any related shortcut are deferred while the active
branch focuses on native Segment Editor wrapping and process recording.

## 9. Audio Recording Infrastructure

`core/_audio_subprocess.py` is the recording worker launched as a subprocess by
`_AudioSubprocess` in the Slicer widget. `core/_audio_recorder.py` provides
standalone chunk-based infrastructure reserved for future local Whisper use.

### 9a. `_AudioSubprocess` (active UI path)

| # | Requirement | Expected behavior |
|---|---|---|
| 9a.1 | Process isolation | Forked with `CREATE_NO_WINDOW` on Windows; stdout/stderr redirected to DEVNULL |
| 9a.2 | Stop protocol | Writes a sentinel file; subprocess drains 150 ms of buffered audio after stop, then writes a result JSON and exits |
| 9a.3 | Timeout | `stop()` waits up to 10 s, then kills the process and calls `wait()` to reap the zombie |
| 9a.4 | Cleanup | `cleanup()` kills process + removes temp directory; called on export completion, widget teardown, and new-recording start |
| 9a.5 | Missing sounddevice | Subprocess exits immediately with error; `start()` still returns; `is_active` becomes False; mouse recording continues unaffected |

### 9b. `StandaloneAudioRecorder` (offline / Whisper path)

| # | Requirement | Expected behavior |
|---|---|---|
| 9b.1 | Import | Does not require microphone dependencies |
| 9b.2 | Start | Lazy-imports `sounddevice` and writes timestamped 16-bit PCM WAV chunks |
| 9b.3 | Ingest | Converts caller-owned frames to PCM16 without using Slicer |
| 9b.4 | Manifest | Writes chunk paths, start/end timestamps, sample rate, channel count, and duration |

## 10. Annotation Summary (TimeLogSummarizer)

`core/TimeLogSummarizer.py` is an offline second-stage summarizer. Input is the
`SegmentHumanBody.annotation_process` JSON; output type is
`SegmentHumanBody.annotation_summary`.

| # | Requirement | Expected behavior |
|---|---|---|
| 10.1 | Input validation | Raises `ValueError` if `type` is not `SegmentHumanBody.annotation_process` |
| 10.2 | Span grouping | Consecutive press/hold/release events are grouped into one `stroke` or `click` span |
| 10.3 | Tool-change folding | A `tool_change` immediately followed by a `press` is folded into the resulting stroke span |
| 10.4 | Point click-place | A `press`/`release`/`place` triplet on the point tool with matching IJK is emitted as a single `point_click_place` span |
| 10.5 | Point drag | Consecutive `point_drag_move` events for the same point are grouped into one `point_drag` span |
| 10.6 | Navigation spans | Consecutive `volume_change` events become `volume_navigation`; consecutive same-view `slice_change` events become `slice_navigation` |
| 10.7 | Carried context | Each span includes the current volume, tool, segment, view, slice, and brush_mm derived from running state |
| 10.8 | Text output | `export_text()` returns one human-readable line per span; strokes with trajectories append an indented trajectory line |
| 10.9 | No Slicer dependency | Runs offline from the saved JSON file without importing Slicer, VTK, or Qt |
| 10.10 | Date header | `export_text()` prepends `Recording: YYYY-MM-DD HH:MM:SS\n` derived from `metadata.start_time` when present |
| 10.11 | Date-change markers | When consecutive spans cross midnight, `--- YYYY-MM-DD ---` is inserted between them; no marker is inserted before the first span |

## 8. Lifecycle

| # | Action | Expected result |
|---|---|---|
| 9.1 | Exit module | Active wrapped tool is disabled |
| 9.2 | Enter module | Parameter node and prompt nodes are rewired |
| 9.3 | Stop recording | Slice-view event filters are removed |
| 9.4 | Scene close | No dangling observers should crash Slicer |
