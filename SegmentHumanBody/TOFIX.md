# TOFIX.md - Native Editor Wrapper Roadmap

This branch keeps Slicer's native editor behavior and builds mouse-centered
process recording around it.

## Completed In This Branch

- Segment lifecycle recording:
  - `segment_removed` from active `vtkSegmentation.SegmentRemoved`
  - `segment_renamed` from active `vtkSegmentation.SegmentModified`
  - segment creation is not recorded as a standalone process event

- Prompt-node wiring that preserves markups placement state.
- Handler wrapper mutual exclusion:
  - Point detach disables point placement.
  - Brush/erase detach disables wrapped Segment Editor effects.
- Segment removal selection:
  - Deleting a segment switches to the immediate previous segment when possible.
- Default model family template:
  - `Default` / `Identity`.
- Mouse-centered recorder scope:
  - Each record gets a sequential `event_id`.
  - Exported top-level schema is `event_id`, `timestamp`, `ras`, `event`,
    `payload`; `ijk` is derived during export and screen-space XY is not
    exported.
  - Records only slice-view events inside active volume.
  - In-volume movement is sampled at 30 records/sec while skipping unchanged
    samples.
  - The recorder is listener-first: raw press/release/move inside volume views
    should be recorded before downstream annotative/non-annotative policy.
  - Slice input is captured with Qt event filters plus VTK interactor observers;
    Qt raw `MouseMove` is suppressed when VTK move capture is available.
  - Wheel/view movement recorded as `view_changed` visual trajectory data.
  - Trajectory events are labeled by kind (`annotation_move`,
    `non_annotation_move`, `view_change`) and role (annotation or visualization
    trajectory).
  - Initial slice visual state cached on one `metadata` event.
  - Record count updated live as records are appended.
  - Point placement recorded semantically as one `point_placed` release
    boundary after interaction end/release.
  - New-point press/release drift is `non_annotation_move` trajectory and does
    not become point relocation.
  - Existing-point relocation records grab/replace boundaries plus
    `non_annotation_move` trajectory.
  - Point-drag move throttling happens before node/RAS work.
  - Point preview/pre-assignment hover movement is not recorded as point drag.
  - Segment selection changes are not standalone events.
  - Parameter-node-to-UI sync has a reentrancy guard to avoid selector feedback
    loops.
## Priority 1 - Validate In Full GUI

These behaviors need manual or full-GUI Slicer validation:

1. Start recording should not change the active Slicer tool.
2. Brush -> Point should show and enable point placement.
3. Point -> Brush/Erase should disable point placement.
4. In-volume mouse trajectory should record at approximately 30 Hz.
5. Raw press/release/move inside volume views should enter the listener event
   stream before annotative/non-annotative policy.
6. Brush/erase held-button movement and released-button hover should both be
   listener-recordable inside volume views.
8. Out-of-volume mouse movement should not record.
9. Moving back into the active volume should resume recording.
10. Point placement should create one `point_placed` boundary after release/end.
11. Existing-point relocation should record grab/replace boundaries and sampled
    `non_annotation_move` movement.


## Priority 2 - Recorder Coverage

The current recorder captures core mouse-centered process data. Remaining gaps:

- Record point deletion semantic events if analysis needs them.
- Record brush parameter changes when users adjust Segment Editor options.
- Record only segmentation-changing hotkeys/actions; avoid general UI-macro
  capture.
- Add full-GUI tests for slice-view event filtering, point placement, and point
  drag events.

## Priority 3 - Model Families

The active template is `Default` / `Identity`. The model-family UI is still
hidden.

Restore path:

1. Keep `Default` / `Identity` as the baseline family.
2. Restore model-family dropdown wiring.
3. Restore model-variant dropdown wiring.
4. Restore `confirmModelSelection`.
5. Reintroduce SPX/SAM/Auto UI buttons through `VISIBLE_BUTTONS`.
6. Keep native Segment Editor brush/erase behavior unchanged.

## Priority 4 - Annotation Log Import/Export

The old TimedMarker annotation-log system is not the same as the current
mouse-centered process recorder.

Before restoring it, decide whether the product needs:

- Annotation-result logs only.
- Annotation-process logs only.
- Both, with separate export formats.

Avoid mixing the two JSON formats unless a migration layer is explicitly added.
