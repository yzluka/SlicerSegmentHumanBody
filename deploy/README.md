# SegmentHumanBody — Setup Guide

This sets up a tool that records how you mark up (segment) medical images in
3D Slicer, plus a second tool that turns any narration you recorded into
text. Two scripts do the work — you copy-paste a couple of commands, wait,
and you're done. You don't need to already have Python, Slicer, or anything
else installed.

Everything you install lives together in one folder you create yourself —
easy to find, easy to move, easy to delete if you ever want a clean slate.

There are two setup scripts, and **both are part of the normal setup** —
the second one isn't a bonus extra. A recording can include your voice
explaining what you're doing; the second script is what turns that into
readable text afterward. If you skip it, that part of your recordings goes
unused.

---

## What you'll need

- A Windows computer and an internet connection.
- About **4 GB** of free disk space.
- No admin/IT permissions needed — everything installs just for your own
  user account.

---

## Step 1 — Create your project folder

1. Anywhere you like (Desktop is fine), right-click → **New → Folder**.
2. Name it something you'll recognize, e.g. **SegmentHumanBody**.
3. Put the three files you got this guide with into it: this file,
   `deploy.ps1`, and `deploy_audio_processor.ps1`.

Everything from here on happens inside this one folder.

---

## Step 2 — Install 3D Slicer into that folder

1. Go to **https://download.slicer.org/** and download the Windows
   installer.
2. Run it. When it asks *where* to install (usually a screen with a
   **Browse** button), choose a location inside the folder you just made —
   for example `...\SegmentHumanBody\Slicer`. Everything else about the
   installer can stay at its defaults.

This is the only manual install in this whole process — everything else
after this is one command at a time.

---

## Step 3 — Set up the recording tool

1. Click the Start menu, type **PowerShell**, and open **Windows
   PowerShell**.
2. Type `cd ` (with a space after it), then drag your project folder into
   the window — it fills in the path for you — and press Enter.
3. Copy-paste this and press Enter:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\deploy.ps1
   ```
4. Wait for it to finish (about a minute). It'll end with a message telling
   you what to do next.

This finds the Slicer you just installed (Step 2), downloads the actual
program files into a new **`SlicerSegmentHumanBody`** folder next to your
scripts (that's expected — leave it there), and creates a
**`Run_SegmentHumanBody.bat`** file in your project folder — that's your
shortcut to open Slicer with the tool ready to go from now on.

---

## Step 4 — Set up the audio-to-text tool

Right after Step 3 finishes, copy-paste this and press Enter:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy_audio_processor.ps1
```

This takes a few minutes — it's doing more work in the background than
Step 3. When it finishes, you'll have a
**`tools\audio_processor\launch.bat`** file inside your project folder,
which opens the text tool.

**If you have an NVIDIA graphics card** and want transcription to run
faster, use this version instead — it also automatically tests that the
speed-up works and tells you if it didn't (in which case it just falls back
to the normal speed, nothing breaks):

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy_audio_processor.ps1 -WithGpu
```

If you're not sure whether you have one of these, it's fine to skip this —
just run the plain version above.

> One small exception to "everything lives in your folder": this step also
> needs a small Python program (about 100 MB) to run the text tool, and that
> one piece installs to Windows' normal per-user apps location instead of
> your project folder, because that turned out to be the more reliable way
> to install it. It's still just for your own account, and everything it
> then builds (the actual tool, a couple GB of it) goes right back into your
> project folder.

---

## Step 5 — Try it

**Record something in Slicer:**

1. Double-click **`Run_SegmentHumanBody.bat`**.
2. In Slicer's search box at the top, type **SegmentHumanBody** and open it.
3. Load an image (`File → Add Data`, or drag a file in).
4. Add a segment and pick a tool (brush, erase, or point) to mark it up with.
5. In the Recording section, check **Audio** and **Mouse+Key** if you want
   to narrate out loud, click **Start/Stop Recording**, mark up the image a
   bit, click stop, then **Export**.

You should now have a JSON file (and a WAV audio file, if you narrated) —
that's your recording.

**Turn that recording into text:**

1. Open **`tools\audio_processor\launch.bat`**.
2. Pick the JSON and WAV files you just exported, then click **Transcribe**.
3. A text file opens up with what it heard — fix anything it got wrong,
   save it.
4. Click **Generate Reports** to get the final text files.

---

## If something goes wrong

**It says something like "running scripts is disabled"**
Make sure you copy-pasted the whole command, including
`-ExecutionPolicy Bypass -File` — don't just double-click the `.ps1` file
itself (Windows opens it as a text file instead of running it).

**It says it can't find a 3D Slicer install**
Go back to Step 2 and make sure the installer finished.

**Step 4 says it can't find something from Step 3**
Run Step 3 again first — Step 4 depends on it.

**Want to start over?**
Delete your whole project folder and start again from Step 1 — Slicer, the
module, and the text tool all live inside it. (The one exception from Step
4 — the small Python program — can be removed separately from Windows'
"Installed Apps" if you want, but it's harmless to leave.)

---

## Advanced options (skip this if the steps above worked)

The rest of this section is for customizing the setup — not needed for a
normal first-time install.

**`deploy.ps1` options:**

| Flag | What it does |
|---|---|
| `-SlicerExe "C:\path\to\Slicer.exe"` | Use a specific Slicer install if it wasn't found automatically |
| `-WithSuperpixelModels` | Also install support for two extra, optional segmentation model choices |

**`deploy_audio_processor.ps1` options:**

| Flag | What it does |
|---|---|
| `-WithGpu` | Attempt and verify NVIDIA GPU-accelerated transcription (see Step 4) |
| `-PythonExe "C:\path\to\python.exe"` | Use a specific Python instead of installing one automatically |

**Technical notes:** The recording tool runs inside 3D Slicer's own built-in
Python, so nothing separate is installed for it. The audio-to-text tool runs
as its own program outside Slicer, so the setup script needs a real Python
to build it from; it checks your project folder and a few standard locations
first, and only downloads and installs one (to Windows' standard per-user
apps folder, not your project folder — see Step 4) if it can't find one
already. Either way, the actual tool and its dependencies end up in
`tools\audio_processor\.venv`, inside your project folder. If GPU mode is
requested, it's verified with a real test transcription rather than assumed
to work, since it's not officially guaranteed to work on Windows; it falls
back to normal (CPU) speed automatically if the test fails.

Both scripts are safe to run again any time — they'll just update what's
already there.
