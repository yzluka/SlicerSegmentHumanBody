# SegmentHumanBody

A 3D Slicer extension for medical image segmentation research. This branch (`annotation-process-recorder`) uses Slicer's native Segment Editor as the editing engine and adds a mouse-centered annotation-process recorder for studying how radiologists annotate images — plus a companion tool that turns any recorded narration into text. Both are covered under [Usage](#usage) below.

## Installation

### Windows (automated, recommended)

`install/windows/` contains two scripts that set up everything on a machine
with nothing preinstalled — no Python, no git:

```powershell
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy.ps1
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy_audio_processor.ps1
```

The first gets 3D Slicer's module running, two ways: drop a downloaded
`Slicer-*-win-amd64.exe` installer next to the script and it installs
Slicer silently for you (verified working), or install Slicer yourself and
the script finds it automatically (`-SlicerExe` if it's somewhere
unusual). The second sets up the audio transcription tool (add `-WithGpu`
if you have an NVIDIA GPU — it installs and *actually verifies* real
GPU-accelerated transcription rather than assuming it works). Both scripts
are part of the normal setup, not an optional extra — a recording's
narration is only useful once the second script can turn it into text.

Full step-by-step walkthrough, troubleshooting, and every script option:
**[`install/windows/README.md`](install/windows/README.md)**.

macOS and Linux installers aren't available yet; use the manual steps below
in the meantime.

### Manual installation (any OS)

```
git clone --branch annotation-process-recorder https://github.com/yzluka/SlicerSegmentHumanBody.git
```

Open 3D Slicer, go to **Modules > Developer Tools > Extension Wizard**, click
**Select Extension**, and choose the repository root folder.

For the audio transcription tool, install its dependencies into any Python
3.9+ environment with `tkinter` available, then run it directly:

```
pip install -r tools/audio_processor/requirements.txt
python -m tools.audio_processor
```

## Usage

### Record an annotation session

1. Open Slicer with the module registered (`Run_SegmentHumanBody.bat` if you
   used the automated installer, or **Modules > Segmentation >
   SegmentHumanBody** if you installed it manually).
2. Load a volume (`File → Add Data`, or drag a file in).
3. Add a segment with the `+` button, and pick a tool — **Brush**, **Erase**,
   or **Point** — to annotate with. These delegate entirely to Slicer's own
   Segment Editor, including undo/redo; there's no custom stroke
   implementation. Handler wrappers enforce mutual exclusion — only one tool
   is active at a time, and deactivating a tool disables the underlying
   Slicer effect.
4. In the Recording section, check **Mouse+Key** and/or **Audio**, then
   click **Start/Stop Recording**. Annotate — optionally **Pause** and
   resume.
5. Click **Start/Stop Recording** again to stop, then **Export** to save
   everything.

### What gets recorded

Recording captures *how* you annotate, not just the final masks, through a
three-stage pipeline:

| Stage | Input | Output | Runs |
|---|---|---|---|
| `MouseEventRecorder` | Live Slicer events | `*_raw.json` | Inside Slicer |
| `TimeLogInterpreter` | `*_raw.json` | `*.json` compact process log | Offline, no Slicer |
| `TimeLogSummarizer` | `*.json` | `*_summary.txt` human-readable spans | Offline, no Slicer |

At 60 Hz: mouse trajectory (device XY) inside the active volume — IJK
derived offline via stored `xy_to_ijk` matrices — button events, tool and
brush-radius changes, segment rename/removal, point placement/relocation/
deletion (from markups events), and volume navigation. Export saves all
three outputs together; a recording can always be re-interpreted offline
from `*_raw.json` alone.

Audio narration is captured via a background subprocess
(`core/_audio_subprocess.py`, `CREATE_NO_WINDOW` on Windows) and starts/stops
together with mouse recording. Pausing freezes mouse recording behind a
modal dialog; resuming records the paused interval. On export, the WAV is
trimmed to the recording start (removing prewarm lead-in) and paused
intervals are silenced (bytes zeroed), with the timeline otherwise
preserved.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `A` / `W` | Next / previous loaded volume sequence (wraps) |
| `Z` / `C` | Previous / next segment (wraps) |
| `Q` | Show/hide current segment |
| `S` | Show/hide other saved segments |

### Turn a recording into text

`tools/audio_processor/` is a standalone GUI (set up by
`deploy_audio_processor.ps1` above, or run manually — see
[Installation](#installation)) that aligns transcribed audio with the
recorded annotation spans:

1. Open it (`tools\audio_processor\launch.bat` on Windows, or
   `python -m tools.audio_processor`).
2. Pick the JSON and WAV files from a recording, choose a model size, and
   click **Transcribe**.
3. Edit the generated `phrases_*.txt` to fix any misheard words, save it.
4. Click **Generate Reports**.

Output:
- `*_transcript.txt` — audio-centric: each phrase aligned with overlapping annotation spans
- `*_caption.txt` — span-centric: each annotation span aligned with overlapping audio
- `*_summary.txt` — human-readable activity log

A `cleaner.py` post-processing layer applies domain-specific text corrections
from JSON pattern files.

### Volume management

- `A` / `W` navigate loaded scalar volumes; segmentation selection stays manual.
- If the selected segmentation's grid doesn't match the new volume, you're
  asked to create a new segmentation, keep the current one, or cancel.
- **Clear Loaded Volumes** removes scalar volumes only; segmentation nodes remain.
- Re-importing a file from the same path replaces the existing node instead
  of creating a `_1` duplicate.

### Model framework

A `Default` / `Identity` model-family template is present for future model
integrations, and SPX superpixel infrastructure remains in the codebase, but
the model-family UI is currently hidden while the recording pipeline is the
active focus.

## Citation

Citation information is not yet available for this work. Please [submit an issue](https://github.com/mazurowski-lab/SlicerSegmentHumanBody/issues) to request a citation.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

If you require a different licensing arrangement, please email the author or [open an issue](https://github.com/mazurowski-lab/SlicerSegmentHumanBody/issues).
