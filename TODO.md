# TODO

Current branch: `feature/native-editor-wrapper`

The current sprint focuses on preserving the existing native Slicer editor UI
while building a robust mouse-centered annotation-process recorder.

## Done

- Native editor wrapper branch established.
- Brush/erase delegate to Slicer's Segment Editor.
- Undo/redo delegate to Slicer's Segment Editor.
- Handler wrappers enforce mutual exclusion.
- Point handler detach disables markups place mode.
- Prompt-node wiring preserves point-placement state.
- Segment deletion selects the immediate previous segment when possible.
- Segment removal and rename events are recorded from active `vtkSegmentation`
  events; segment creation is not recorded.
- Segment switching is not recorded as a standalone event.
- Brush/erase drag samples or releases infer a missing `press` boundary when
  Slicer drops the initial press event.
- Mouse recorder captures only in-volume slice-view events.
- Mouse recorder listens to both Qt slice-view events and VTK interactor
  events for native Segment Editor brush/erase capture; VTK observers are
  high-priority and non-consuming so Paint/Erase cannot hide brush events first.
- Compact export uses event `id` values starting at 1 inside
  `{type, metadata, events}`; metadata is not an event.
- Exported records use RAS as the canonical coordinate; IJK is derivable from
  RAS plus metadata and screen-space XY is not exported.
- Session metadata caches initial Red/Green/Yellow slice visual state; later
  visual snapshots are stored only for explicit view-change events and omit
  repeated slice-view dimensions.
- Record status count updates as events are appended.
- In-volume movement is sampled at 30 records/sec.
- Brush/erase press, release, held-button movement, and released-button hover
  are listener-recordable inside volume views.
- Wheel/view movement is recorded as visual `view_changed` trajectory data.
- Trajectories are labeled by kind (`annotation_move`, `non_annotation_move`, `view_change`) and role (annotation vs visualization trajectory).
- Point placement is recorded as one semantic `point_placed` verdict with point
  ID/name.
- New-point press/release drift is treated as `non_annotation_move`, not point relocation.
- Existing-point relocation records grab, sampled `non_annotation_move`
  trajectory, and final `point_placed` replace.
- Point deletion records `point_removed` with the last cached point location.
- Point-drag movement is throttled before node/RAS work.
- Point preview/pre-assignment hover movement is excluded from point-drag data.
- Parameter-node-to-UI sync is guarded against selector feedback loops.
- Added `Default` / `Identity` model-family template.

## Next

## Priority

1. Multi-volume switching and manual segmentation control.

   When multiple volumes are loaded, switching the active volume should also
   switch the slice views to that volume. Segmentation selection is manual:
   volume switching should not auto-match, clear, or switch the
   selected segmentation. Radiology folders may contain different sequences
   with different identifiers but only one intended segmentation. If the
   selected segmentation voxel grid shape, spacing, or orientation differs from
   the target volume, ask whether to create a new empty segmentation for that
   volume, keep the current segmentation, or cancel the switch. Origin
   differences are ignored for this compatibility check. If a compatible target
   volume has a zero origin, copy the first compatible non-zero scene origin
   onto that target during volume import scanning and before display so native
   Slicer offsets line up. If the user explicitly creates a segment while no
   segmentation node exists, create a generic segmentation for the current
   volume. Record the selected volume, segmentation, and segment pair explicitly
   instead of relying on filename conventions.

   The active workflow assumes the user drags one folder for one patient.
   After import, scan all loaded scalar volumes as one set. Shape, spacing, and
   orientation must be consistent. Origin differences are allowed and
   normalized. Inconsistent spacing/shape/orientation should show one
   informational warning after the import stream settles, with statistics and
   filenames for all imported volume geometry groups, but should not prevent
   loading or sequence switching.

   Recording metadata should include all loaded/involved volume sequences with
   sequence index, node ID, and name. Volume switches should appear in the raw
   log and as compact `volume_change` events in the semantic log.

   Provide a `Clear Loaded Volumes` button that removes scalar volume nodes
   only. Segmentation nodes remain manually controlled.

   Re-importing the same volume file should replace the older scalar volume
   node and use the full storage filename, including suffix such as `.nii.gz`,
   instead of keeping Slicer's `_1` duplicate.

2. Sequence navigation hotkeys.

   Add hotkeys for moving between volume sequences:

   - `A`: next loaded volume sequence
   - `W`: previous loaded volume sequence
   - Switching wraps around at the first/last sequence.

3. Rebind editing/view hotkeys.

   - `Z`: previous segment
   - `C`: next segment
   - Segment switching wraps around at the first/last segment.
   - `Q`: show/hide current segment
   - `S`: show/hide other saved segments
   - `E`: reserved for future use
   - No shortcut for segment creation.
   - Superpixel-grid shortcut/functionality is deferred for now.

1. Full-GUI recording validation.

   Verify manually in Slicer:

   - Start recording does not switch to point placement.
   - Brush/erase/point mutual exclusion works visually.
   - In-volume movement records at about 30 Hz.
   - Brush/erase held movement and released hover movement both appear in the
     listener event stream before downstream policy filtering.
   - Out-of-volume movement does not record.
   - Point placement records `point_placed`.

2. Recorder event completeness.

   Add semantic events for:

   - Brush radius changes.
   - Other Segment Editor option changes that affect mask edits.

3. Tests.

   Add full-GUI Slicer tests for:

   - Slice-view-only event filters.
   - In-volume/out-of-volume event filtering.
   - Markups `point_placed` recording.
   - Markups point-drag start/move/end recording.
   - Handler wrapper visual state.

4. Model families.

   Keep `Default` / `Identity` as the template. Restore model-family UI wiring
   only after recorder behavior is stable.

## Older Work To Revisit Later

The TimedMarker annotation-log model and SPX/SAM/Auto model-family systems still
exist in code, but they are not the current sprint focus. Revisit them after the
mouse-centered process recorder is stable.

SPX capability should be re-implemented/restored later. The remaining SPX
implementation in `SegmentHumanBody/core/modelFamilies.py`,
`SegmentHumanBody/core/models/spx.py`, `SegmentHumanBody/core/_logic.py`, and
related tests should be treated as reference and reusable infrastructure rather
than discarded legacy code.
