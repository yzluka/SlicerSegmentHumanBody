# Windows Installation

Two scripts set up everything on a Windows machine with nothing preinstalled
— no Python, no git needed. Run them from the repo root (a PowerShell
window opened in this repo's top-level folder):

```powershell
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy.ps1
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy_audio_processor.ps1
```

Both are required — the second one isn't a bonus extra. A recording can
include your voice explaining what you're doing; the second script is what
turns that into readable text afterward. If you skip it, that part of your
recordings goes unused.

(macOS/Linux support isn't available yet — see [`../`](../) for other
platforms as they're added, or the manual cross-platform instructions in the
main [README](../../README.md#installation) for now.)

---

## Before you start

- An internet connection. About **4 GB** free disk space (~1.5 GB for 3D
  Slicer, ~2.5 GB for the audio tool).
- No admin/IT permissions needed — everything installs just for your own
  user account.
- If you don't have this repo yet: copy just `deploy.ps1` and
  `deploy_audio_processor.ps1` (from this folder) to an empty folder of your
  own and run them from there instead — the first script will fetch this
  repo automatically rather than expecting it to already be present.

---

## Step 1 — Install 3D Slicer

Go to **https://download.slicer.org/**, download the Windows installer, and
run it (default options are fine). `deploy.ps1` looks for it in the usual
install locations automatically, so it doesn't matter exactly where — the
repo root is a reasonable choice if you don't already have a preference.

---

## Step 2 — Set up the recording module

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy.ps1
```

This finds the Slicer you just installed and creates
**`Run_SegmentHumanBody.bat`** at the repo root — your shortcut to open
Slicer with the module ready to go from now on.

Optional: add `-WithSuperpixelModels` to also install support for two extra
segmentation model choices (SLIC / Felzenszwalb) — not needed for the
default recorder.

---

## Step 3 — Set up the audio-to-text tool

```powershell
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy_audio_processor.ps1
```

This takes a few minutes — it's doing more work in the background than
Step 2. When it finishes, you'll have **`tools\audio_processor\launch.bat`**,
which opens the text tool.

**If you have an NVIDIA graphics card** and want transcription to run
faster, use this version instead — it also automatically tests that the
speed-up works and tells you if it didn't (in which case it just falls back
to the normal speed, nothing breaks):

```powershell
powershell -ExecutionPolicy Bypass -File .\install\windows\deploy_audio_processor.ps1 -WithGpu
```

If you're not sure whether you have one, it's fine to skip this — just run
the plain version above.

> One exception to "everything lives in the repo": this step also needs a
> small Python program (about 100 MB) to run the text tool, and that one
> piece installs to Windows' normal per-user apps location rather than into
> the repo, because that turned out to be the more reliable way to install
> it. It's still just for your own account, and everything it then builds
> (the actual tool, a couple GB of it) goes into
> `tools\audio_processor\.venv`, inside the repo.

---

## If something goes wrong

**"running scripts is disabled on this system"**
Make sure you copy-pasted the whole command, including
`-ExecutionPolicy Bypass -File` — don't just double-click the `.ps1` file
itself (Windows opens it as a text file instead of running it).

**It says it can't find a 3D Slicer install**
Go back to Step 1 and make sure the installer finished, or pass
`-SlicerExe "C:\path\to\Slicer.exe"`.

**Step 3 says it can't find something from Step 2**
Run Step 2 again first — Step 3 depends on the module source it fetches.

**GPU acceleration "did NOT verify"**
Your GPU/driver combo isn't cooperating with the pip-installed CUDA
libraries (a known rough edge on Windows, not just for this tool).
Everything still works on CPU — the Device dropdown's `auto` detects this
automatically. For guaranteed GPU speed on Windows, see
[Purfview's whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win)
as an alternative.

**Want to start over?**
Delete `Run_SegmentHumanBody.bat`, `tools\audio_processor\launch.bat`, and
`tools\audio_processor\.venv\`, then re-run Steps 2–3. (The small Python
program from Step 3 can be removed separately from Windows' "Installed
Apps" if you want, but it's harmless to leave.)

---

## Advanced options

**`deploy.ps1`:**

| Flag | What it does |
|---|---|
| `-SlicerExe "C:\path\to\Slicer.exe"` | Use a specific Slicer install if it wasn't found automatically |
| `-RepoUrl` / `-Branch` / `-RepoDir` | Override where/what to fetch (irrelevant when already run from inside a checkout, as here) |
| `-WithSuperpixelModels` | Also install support for two extra, optional segmentation model choices |

**`deploy_audio_processor.ps1`:**

| Flag | What it does |
|---|---|
| `-WithGpu` | Attempt and verify NVIDIA GPU-accelerated transcription (see Step 3) |
| `-PythonExe "C:\path\to\python.exe"` | Use a specific Python instead of installing one automatically |

**Technical notes:** Both scripts detect when they're already running from
inside a checkout of this repo (as here) and use it directly instead of
downloading a redundant copy — deliberately without touching its git state
(no `fetch`/`reset`), so a working copy with local changes is never reset.
The recording module runs inside 3D Slicer's own built-in Python, so nothing
separate is installed for it. The audio-to-text tool runs as its own program
outside Slicer, so it needs a real Python to build a virtual environment
from; the script checks the repo and a few standard locations first, and
only downloads and installs one if it can't find one already. If GPU mode is
requested, it's verified with a real test transcription rather than assumed
to work, since it's not officially documented as supported on Windows.

Both scripts are safe to run again any time — they'll just update what's
already there.
