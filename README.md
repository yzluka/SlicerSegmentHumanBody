# SegmentHumanBody

A 3D Slicer extension for medical image segmentation research. This branch (`annotation-process-recorder`) uses Slicer's native Segment Editor as the editing engine and adds a mouse-centered annotation-process recorder for studying how radiologists annotate images.

## What It Does

### Interactive Annotation

Brush, erase, and prompt-point placement delegate entirely to Slicer's built-in Segment Editor. Undo/redo uses Slicer's native undo stack. No custom stroke implementation.

Handler wrappers enforce mutual exclusion — only one tool is active at a time, and deactivating a tool disables the underlying Slicer effect.

### Annotation-Process Recording

Records *how* annotators work, not just the final masks.

**Three-stage pipeline:**

| Stage | Input | Output | Runs |
|---|---|---|---|
| `MouseEventRecorder` | Live Slicer events | `*_raw.json` | Inside Slicer |
| `TimeLogInterpreter` | `*_raw.json` | `*.json` compact process log | Offline, no Slicer |
| `TimeLogSummarizer` | `*.json` | `*_summary.txt` human-readable spans | Offline, no Slicer |

What is recorded at 60 Hz:
- Mouse trajectory (device XY) inside the active volume — IJK derived offline via stored `xy_to_ijk` matrices
- Button events, tool changes, brush radius changes
- Segment rename and removal
- Point placement, relocation, and deletion (from markups events)
- Volume navigation and slice changes
- Audio narration (optional, via background subprocess)

Pressing **Export** saves all three outputs alongside each other. Recordings can be re-interpreted offline from `*_raw.json` alone.

### Audio Recording

Integrated microphone capture via a background subprocess (`core/_audio_subprocess.py`, CREATE_NO_WINDOW on Windows). Audio and mouse recording start and stop together.

**Pause/Resume:** Pause freezes mouse recording and shows a modal dialog. Resuming records the paused interval. On export, the WAV is trimmed to the recording start (removing prewarm lead-in) and paused intervals are silenced (bytes zeroed) while the timeline is preserved.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` / `W` | Next / previous loaded volume sequence (wraps) |
| `Z` / `C` | Previous / next segment (wraps) |
| `Q` | Show/hide current segment |
| `S` | Show/hide other saved segments |

### Offline Analysis Tool

`tools/audio_processor/` is a standalone GUI for aligning transcribed audio with recorded annotation spans. Given a recording's WAV and JSON files, it produces:

- `*_caption.txt` — span-centric view: each annotation span aligned with overlapping audio
- `*_transcript.txt` — audio-centric view: each audio phrase aligned with overlapping annotation spans
- `*_summary.txt` — human-readable activity log

Includes a post-processing layer (`cleaner.py`) for applying domain-specific text corrections from JSON pattern files.

## Installation

Clone the repository:

```
git clone https://github.com/mazurowski-lab/SlicerSegmentHumanBody.git
```

Open 3D Slicer, go to **Modules > Developer Tools > Extension Wizard**, click **Select Extension**, and choose the repository root folder.

### Offline analysis tool (`tools/audio_processor/`)

Install dependencies into your Python environment:

```
pip install -r tools/audio_processor/requirements.txt
```

To launch via the provided batch file on Windows, edit `tools/audio_processor/launch.bat` first — it contains hardcoded paths to a specific machine. Update the `PYTHONPATH` and `pythonw.exe` lines to match your Python installation before running it. Alternatively, run the tool directly:

```
python -m tools.audio_processor
```

## Basic Usage

1. Load a volume in 3D Slicer.
2. Open **Modules > Segmentation > SegmentHumanBody**.
3. Add segments with the `+` button.
4. Use **Brush**, **Erase**, or **Point** tools to annotate.
5. Check **Mouse+Key** and/or **Audio**, then click **Record**.
6. Annotate. Optionally **Pause** and resume.
7. Click **Record** again to stop, then **Export** to save all files.

## Volume Management

- `A` / `W` navigate loaded scalar volumes; segmentation selection stays manual.
- If the selected segmentation's grid does not match the new volume, you are asked to create a new segmentation, keep the current one, or cancel.
- **Clear Loaded Volumes** removes scalar volumes only; segmentation nodes remain.
- Re-importing a file from the same path replaces the existing node instead of creating a `_1` duplicate.

## Model Framework

A `Default` / `Identity` model-family template is present for future model integrations. SPX superpixel infrastructure remains in the codebase. The model-family UI is currently hidden while the recording pipeline is the active focus.

## Citation

Citation information is not yet available for this work. Please [submit an issue](https://github.com/mazurowski-lab/SlicerSegmentHumanBody/issues) to request a citation.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

If you require a different licensing arrangement, please email the author or [open an issue](https://github.com/mazurowski-lab/SlicerSegmentHumanBody/issues).

