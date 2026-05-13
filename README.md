# SegmentHumanBody

[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.12974-orange.svg?style=flat)](https://arxiv.org/abs/2401.12974)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2505.01854-orange.svg?style=flat)](https://www.arxiv.org/abs/2505.01854)

SegmentHumanBody aims to asist its users in segmenting medical data on <a href="https://github.com/Slicer/Slicer">3D Slicer</a> by integrating the <a href="https://github.com/mazurowski-lab/SegmentAnyBone">SegmentAnyBone</a>, and <a href="https://github.com/mazurowski-lab/SLM-SAM2">SLM-SAM2</a> developed by Mazurowski Lab.

## Current Development Branch

The active development branch (`feature/native-editor-wrapper`) keeps the
existing Slicer module UI but delegates interactive editing to Slicer's native
Segment Editor tools. Brush and erase use Slicer's Paint/Erase effects, prompt
points use native markups nodes, and undo/redo use Slicer's native undo stack.

The current focus is a mouse-centered annotation-process recorder:

- records only Red/Green/Yellow slice-view mouse events inside the active volume;
- stores raw device XY per event; IJK coordinates are derived offline by `TimeLogInterpreter` using stored per-view `xy_to_ijk` matrices;
- assigns exported process events compact `id` values starting at 1; metadata is not an event;
- exports a compact `{type, metadata, events}` process log;
- uses IJK as the canonical compact coordinate; point events carry world RAS from Slicer markups, converted to IJK via stored `ras_to_ijk`;
- samples in-volume movement at 60 Hz, then adaptively thins by cached XY-to-IJK scale;
- listens to both Qt slice-view events and high-priority, non-consuming VTK slice interactor events so native Segment Editor brush/erase input is captured before Paint/Erase can consume it;
- records mouse status (`move`, `press`, `release`, `view`);
- records active handler/tool plus only the parameters needed for reconstruction, such as brush radius;
- differentiates `annotation_move` from `non_annotation_move`, with matching `annotation_trajectory` or `visualization_trajectory` roles;
- caches initial Red/Green/Yellow slice views at recording start and stores later visual snapshots only for explicit view-change events, without repeated slice-view dimensions;
- updates the recording event count as records arrive;
- records new point placement as one semantic `point_placed` verdict with point ID/name while keeping raw point listener boundaries recordable;
- treats press-release drift during new point placement as `non_annotation_move` trajectory;
- keeps segment switches out of the event stream, relying on the segment ID carried by boundary, trajectory, and semantic events;
- infers a brush/erase `press` boundary if Slicer drops the initial press but a drag sample or release is observed;
- records brush/erase press, release, held-button movement, and released-button hover as listener events inside the active volume;
- records segment removal and segment rename, but not segment creation;
- records existing-point relocation as `point_drag_start` grab, sampled `non_annotation_move` trajectory, and final `point_replaced` replace;
- records control-point deletion as `point_removed` with the last cached point location;

Current module shortcuts:

- `A` / `W`: next / previous loaded volume sequence.
- `Z` / `C`: previous / next segment.
- `Q`: show/hide the current segment.
- `S`: show/hide saved/other segments.
- `E`: reserved for future use.

Volume and segment shortcuts wrap around at the ends. Volume sequence switching
updates the source volume and slice views only.
Segmentation selection is manual; switching sequences does not auto-match,
auto-match, clear, or switch segmentation nodes. If the selected segmentation
has a different voxel grid shape, spacing, or orientation than the target
volume, the module asks whether to create a new empty segmentation for that
volume, keep the current segmentation, or cancel the switch. Origin differences
are ignored for this compatibility check. When a compatible target volume has a
zero origin, the module copies the first compatible non-zero scene origin onto
that volume. This scan runs when volumes are imported and again before display,
so native Slicer slice offsets line up across CT and derived sequences. When a
segment is explicitly created and no segmentation exists, the module creates a
generic segmentation node for the current volume. The process recorder stores
the explicit selected volume, segmentation, and segment identities instead of
relying on filename conventions.

The current workflow assumes a dragged folder contains one patient's compatible
volume set. All loaded scalar volumes are checked as a group after import:
shape, spacing, and orientation must match. Origin may differ and is normalized.
If spacing/shape/orientation differs, the module shows one informational
warning after the import stream settles, listing geometry-group statistics and
filenames for all loaded volumes. Loading and sequence switching are still
allowed.

Recordings store the loaded volume sequence list in metadata, including each
sequence index, node ID, and name. Volume switches are kept in the raw log and
interpreted into compact `volume_change` events in the semantic annotation log.

Use `Clear Loaded Volumes` to remove all scalar volume nodes from the scene.
Segmentation nodes are intentionally left untouched for manual deletion.
If a volume file is imported again from the same path, the newly loaded node
replaces the older scalar volume node and uses the full storage filename,
including suffix such as `.nii.gz`, instead of leaving a `_1` duplicate.

Superpixel-grid display is deferred for now. The SPX implementation remains in
the codebase as reusable infrastructure for later restoration.

A second-stage summarizer, `SegmentHumanBody/core/TimeLogSummarizer.py`, converts
the compact `annotation_process` log into a human-readable `annotation_summary`
with higher-level activity spans (`stroke`, `click`, `volume_navigation`,
`point_click_place`, `point_drag`, etc.), each carrying full volume/tool/segment
context. Both `TimeLogInterpreter` and `TimeLogSummarizer` run offline with no
Slicer dependency.

A standalone audio recorder is available in `SegmentHumanBody/core/_audio_recorder.py`
for future local Whisper transcription workflows. It records timestamped
Whisper-friendly WAV chunks and metadata, but it is not connected to the module
UI or process recorder yet.

The model-family framework is still present. A placeholder `Default` family with
an `Identity` variant is available as a template for future model integrations.

## License

The repository is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Installation via GitHub Repository

You can clone this repository by running the following command:

```
git clone https://github.com/mazurowski-lab/SlicerSegmentHumanBody.git
```

After cloning the repository, you need to introduce the extension to 3D Slicer. Please go to Modules > Developer Tools > Extension Wizard on 3D Slicer and click 'Select Extension' button. You should select the root folder that contains this repository in the pop-up. If you don't get any error on Python terminal, that means you are ready to use the extension!

## Usage

### Preparation

- Load an image to be segmented.
- Choose **Modules > Segmentation > SegmentHumanBody** module in the module selector. The first time the module is started, it will ask permission to install pytorch. Give permission and wait several minutes for the installation to complete. The application will not be responsive during the installation.
- Click "Configure labels in the segment editor" button to create a label for each structure that will be segmented. For example, "hip" and "femur".
- Go back to **SegmentHumanBody** module by clicking the green left arrow on the module toolbar. You are ready to segment now!

### Automatic Segmentation

<img src="Screenshots/sws1.png" width=45%> <img src="Screenshots/sws2.png" width=42%>

- Click "Run Automatic Segmentation" button and wait until the processing is completed (it may take some time based on number of slices). SegmentAnyBone will run in automated mode and will segment bones in each slice it can detect. This is going to be a binary segmentation (e.g. bone vs no-bone). To assign labels to different object of interests in slices, you can use label assignment feature: 
- For each structure of interest: choose a label to assign (above the "Run automatic segmentation" button, for example: "hip") and click "Assign label (2D)" or "Assign label (3D)" button
  - **2D Label Assignment:** Change label of the connected component *only in the current slice*. You should first place a prompt point on the object of interest whose label you want to change and click Assign Label (2D) button. Label of the object of interest will be changed only on that particular slice.
  - **3D Label Assignment:** Change label of the connected component *through consecutive slices in 3D*. You should first place a prompt point on the object of interest whose label you want to change. Then, you should click Assign Label (3D) button. Label of the connected component through consecutive slices will be changed.

### Prompt Based Segmentation

<img src="Screenshots/sws3.png" width=45% height=45%>

- Select the label you want to segment from the dropdown list (for example "femur" as shown in the image above).
- Click "Start Segmentation for Current Slice" button.

If it is the first time to segment a slice of this file, you need to wait for SegmentHumanBody to produce some files that will be used for the segmentation. After SegmentHumanBody generated these files, you can start putting **prompt points** or **prompt boxes** on the current slice. You'll be able to see the segmentation mask on 3D Slicer. Please click "Stop Segmentation for Current Slice" whenever you finish your segmentation for the current slice.

If you are not satisfied with the segmentation mask produced by SegmentHumanBody, you can edit it as you wish using the "Segment Editor" module of 3D Slicer.

## Future Work

More segmentation models that aims to segment various type of tissues will be added in the future.

## Citation

If you find our work to be useful for your research, please cite [our paper](https://arxiv.org/abs/2401.12974):

```bibtex
@misc{gu2024segmentanybone,
      title={SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI},
      author={Hanxue Gu and Roy Colglazier and Haoyu Dong and Jikai Zhang and Yaqian Chen and Zafer Yildiz and Yuwen Chen and Lin Li and Jichen Yang and Jay Willhite and Alex M. Meyer and Brian Guo and Yashvi Atul Shah and Emily Luo and Shipra Rajput and Sally Kuehn and Clark Bulleit and Kevin A. Wu and Jisoo Lee and Brandon Ramirez and Darui Lu and Jay M. Levin and Maciej A. Mazurowski},
      year={2024},
      eprint={2401.12974},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
```
