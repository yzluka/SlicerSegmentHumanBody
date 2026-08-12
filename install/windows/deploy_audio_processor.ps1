<#
.SYNOPSIS
  Deploys the audio_processor tool (tools/audio_processor) from the
  SegmentHumanBody repo -- the second half of the standard deployment,
  alongside deploy.ps1. It is a standalone tkinter GUI that transcribes
  recorded WAV/JSON pairs with faster-whisper into text reports. It runs as
  its own process, separate from Slicer, which is why it needs its own
  Python (see below) -- but it is treated as required, not a skippable
  extra: run deploy.ps1 first (it fetches the module source this script
  needs), then this script.

.DESCRIPTION
  Unlike the Slicer module, this tool cannot run on Slicer's bundled Python:
  it needs tkinter (Slicer's embedded Python does not ship Tcl/Tk) and it
  installs its own package stack (faster-whisper -> ctranslate2, onnxruntime,
  huggingface-hub, av, tokenizers). So on a machine with "no Python", this
  script:

    1. Looks for a standalone CPython 3.9-3.13 (Slicer's PythonSlicer.exe is
       deliberately NOT considered -- it lacks tkinter and mixing this tool's
       heavy deps into Slicer's own env is not worth the risk).
    2. If none is found, downloads and silently installs a pinned CPython
       3.12.10 **per-user** (no admin rights required) from python.org.
       3.12 is pinned deliberately: ctranslate2 (faster-whisper's backend)
       does not yet publish Windows wheels for CPython 3.14, so installing
       "whatever is newest" can leave pip unable to install faster-whisper
       at all.
    3. Creates an isolated venv under tools/audio_processor/.venv (keeps
       these deps off both Slicer's Python and any other Python on the
       machine).
    4. pip installs tools/audio_processor/requirements.txt (CPU-capable by
       default -- this always works and needs no GPU).
    5. Optionally (-WithGpu), attempts CUDA acceleration: installs the NVIDIA
       cuBLAS/cuDNN pip wheels (NOT torch -- CTranslate2, faster-whisper's
       actual inference backend, makes its own direct CUDA calls and doesn't
       use torch at all; torch would only add ~490MB to install the one
       device="auto" convenience check in _transcribe.py). Then runs a REAL
       transcription on device=cuda and reports whether it worked -- loading
       the model object alone doesn't prove GPU inference works, since
       CTranslate2 only loads cuBLAS/cuDNN lazily on the first forward pass.
       faster-whisper's own docs only document the pip cuBLAS/cuDNN trick for
       Linux; on Windows this is best-effort, so the script verifies rather
       than assumes, and tells you what to do if it fails.
    6. Regenerates launch.bat pointing at the venv's own pythonw.exe (the
       committed launch.bat is hardcoded to one developer's machine and will
       not work elsewhere as-is).

.PARAMETER RepoDir
  Path to the SlicerSegmentHumanBody checkout. Auto-detected the same way as
  deploy.ps1: if this script is already sitting inside a checkout of the
  repo, that's used directly; otherwise defaults to .\SlicerSegmentHumanBody
  next to this script.

.PARAMETER PythonExe
  Use this specific python.exe as the base interpreter instead of auto-detecting
  or installing one. Must be CPython 3.9-3.13 with tkinter.

.PARAMETER WithGpu
  Also attempt CUDA GPU acceleration (requires an NVIDIA GPU + current driver).
  Falls back to reporting failure clearly if it doesn't pan out -- CPU mode
  keeps working either way.

.EXAMPLE
  .\deploy_audio_processor.ps1
  .\deploy_audio_processor.ps1 -WithGpu
#>

[CmdletBinding()]
param(
    [string]$RepoDir = "",
    [string]$PythonExe = "",
    [switch]$WithGpu
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Err2($msg)  { Write-Host "    $msg" -ForegroundColor Red }

# Walks up from $startDir looking for a folder that looks like the repo root
# (has SegmentHumanBody/SegmentHumanBody.py). Returns $null if not found
# within a few levels. Mirrors deploy.ps1's Find-RepoRoot -- not a fixed
# relative offset, since this script may live at install/windows/ today and
# move again later (e.g. once install/mac, install/linux exist).
function Find-RepoRoot([string]$startDir) {
    $dir = $startDir
    for ($i = 0; $i -lt 5; $i++) {
        if (Test-Path (Join-Path $dir "SegmentHumanBody\SegmentHumanBody.py")) { return $dir }
        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    return $null
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DetectedRepoRoot = Find-RepoRoot $ScriptDir
if ([string]::IsNullOrWhiteSpace($RepoDir)) {
    if ($DetectedRepoRoot) {
        $RepoDir = $DetectedRepoRoot
    } else {
        $RepoDir = Join-Path $ScriptDir "SlicerSegmentHumanBody"
    }
}
$ToolDir = Join-Path $RepoDir "tools\audio_processor"
$ReqFile = Join-Path $ToolDir "requirements.txt"

if (-not (Test-Path $ReqFile)) {
    throw "Can't find $ReqFile. Run deploy.ps1 first to fetch the repo, then re-run this script."
}

# ---------------------------------------------------------------------------
# 1. Find (or install) a standalone CPython 3.9-3.13 with tkinter.
#    Slicer's PythonSlicer.exe is intentionally not considered here.
# ---------------------------------------------------------------------------
Write-Step "Looking for a standalone Python (3.9-3.13, with tkinter)"

function Test-CandidatePython([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    try {
        $verOut = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $verOk = $verOut -match '^3\.(9|1[0-3])$'
        if (-not $verOk) { return $false }
        & $exe -c "import tkinter" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$baseExe = ""
if ($PythonExe) {
    if (Test-CandidatePython $PythonExe) {
        $baseExe = $PythonExe
    } else {
        throw "-PythonExe '$PythonExe' is missing, not CPython 3.9-3.13, or lacks tkinter."
    }
}

if (-not $baseExe) {
    $searchRoots = @(
        $ScriptDir,
        (Join-Path $env:LocalAppData "Programs\Python"),
        "$env:ProgramFiles\Python*",
        "${env:ProgramFiles(x86)}\Python*"
    )
    if ($DetectedRepoRoot) { $searchRoots = @($DetectedRepoRoot) + $searchRoots }
    $found = @()
    foreach ($root in $searchRoots) {
        $found += Get-ChildItem -Path $root -Filter "python.exe" -Recurse -Depth 1 -ErrorAction SilentlyContinue
    }
    foreach ($cmdName in @("python", "python3")) {
        $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue
        if ($cmd) { $found += Get-Item $cmd.Source }
    }
    foreach ($cand in ($found | Select-Object -ExpandProperty FullName -Unique)) {
        if (Test-CandidatePython $cand) { $baseExe = $cand; break }
    }
}

if (-not $baseExe) {
    Write-Warn2 "No usable standalone Python found -- installing CPython 3.12.10 (per-user, no admin needed)."
    Write-Warn2 "This one piece installs to Windows' standard per-user apps location rather than"
    Write-Warn2 "inside this project folder -- pointing it at a custom folder turned out to be an"
    Write-Warn2 "unreliable silent-install path. Everything built from it (the actual tool and its"
    Write-Warn2 "~2+ GB of packages) still lives entirely inside $RepoDir\tools\audio_processor\.venv."
    $pyVersion = "3.12.10"
    $installerUrl = "https://www.python.org/ftp/python/$pyVersion/python-$pyVersion-amd64.exe"
    $installerPath = Join-Path $env:TEMP "python-$pyVersion-amd64.exe"

    Write-Ok "Downloading $installerUrl"
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

    Write-Ok "Installing silently (per-user)..."
    $installArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "Include_tcltk=1",
        "Include_pip=1"
    )
    $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
    Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        throw "Python installer exited with code $($proc.ExitCode)."
    }

    $expected = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
    if (Test-Path $expected) {
        $baseExe = $expected
    } else {
        $fallback = Get-ChildItem -Path (Join-Path $env:LocalAppData "Programs\Python") `
            -Filter "python.exe" -Recurse -Depth 1 -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $fallback) { throw "Python installed but python.exe could not be located afterward." }
        $baseExe = $fallback.FullName
    }

    if (-not (Test-CandidatePython $baseExe)) {
        throw "Installed Python at $baseExe but it failed the tkinter/version check."
    }
    Write-Ok "Installed: $baseExe"
} else {
    Write-Ok "Using existing Python: $baseExe"
}

# ---------------------------------------------------------------------------
# 2. Create an isolated venv for this tool
# ---------------------------------------------------------------------------
Write-Step "Setting up an isolated virtual environment"

$VenvDir = Join-Path $ToolDir ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$VenvPyw = Join-Path $VenvDir "Scripts\pythonw.exe"

if (-not (Test-Path $VenvPy)) {
    & $baseExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)." }
    Write-Ok "Created venv at $VenvDir"
} else {
    Write-Ok "Reusing existing venv at $VenvDir"
}

# ---------------------------------------------------------------------------
# 3. Install requirements (CPU-capable baseline -- always works)
# ---------------------------------------------------------------------------
Write-Step "Installing tools/audio_processor/requirements.txt (this pulls in ctranslate2, onnxruntime, etc. -- may take a few minutes)"

& $VenvPy -m pip install --upgrade --quiet pip
& $VenvPy -m pip install --quiet -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit $LASTEXITCODE)." }
Write-Ok "faster-whisper and its dependencies installed."

# ---------------------------------------------------------------------------
# 4. Optional: attempt GPU (CUDA) acceleration
# ---------------------------------------------------------------------------
$gpuVerified = $false
$gpuPathExtra = ""

if ($WithGpu) {
    Write-Step "Attempting GPU acceleration (best-effort on Windows)"
    Write-Warn2 "faster-whisper's own docs only document the pip cuBLAS/cuDNN trick for"
    Write-Warn2 "Linux; on Windows this is not officially covered. This step installs the"
    Write-Warn2 "pieces, then runs a REAL transcription on the GPU to confirm it actually"
    Write-Warn2 "works instead of assuming it does. Requires an NVIDIA GPU + current driver."

    # Deliberately NOT installing torch. faster-whisper's actual inference
    # backend is CTranslate2, a self-contained compiled engine that makes its
    # own direct CUDA/cuBLAS/cuDNN calls -- verified by running real GPU
    # transcription below with no torch present at all. torch is a ~490MB
    # download that, in this codebase, is only ever consulted for the
    # device="auto" convenience check in _transcribe.py (falls back to cpu
    # if torch isn't importable) -- it plays no role in whether GPU
    # transcription itself works. CTranslate2 exposes the same "is there a
    # CUDA device" information for free via ctranslate2.get_cuda_device_count().
    & $VenvPy -m pip install --quiet "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "nvidia-cublas-cu12/nvidia-cudnn-cu12 pip install failed (no prebuilt wheel for"
        Write-Warn2 "this Python/platform combo is common). Continuing -- torch's own bundled CUDA"
        Write-Warn2 "libraries may still be enough for the verification step below."
    }

    # Locate the DLL directories pip may have just installed, so the
    # launcher can put them on PATH (Windows resolves dependent DLLs via
    # PATH; there is no LD_LIBRARY_PATH equivalent here).
    $sitePkgs = & $VenvPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $dllDirs = @()
    foreach ($pkg in @("nvidia\cublas\bin", "nvidia\cudnn\bin")) {
        $p = Join-Path $sitePkgs $pkg
        if (Test-Path $p) { $dllDirs += $p }
    }
    $gpuPathExtra = ($dllDirs -join ";")

    Write-Ok "Running a REAL transcription on device=cuda to verify (downloads the 'tiny' model, needs internet)..."
    Write-Ok "(Constructing the model object alone is not a sufficient check -- CTranslate2 loads"
    Write-Ok "cuBLAS/cuDNN lazily, on the first actual encode() forward pass, not at construction.)"
    $verifyScript = Join-Path $env:TEMP "shb_gpu_verify.py"
    @'
import sys
try:
    import numpy as np
    from faster_whisper import WhisperModel
    m = WhisperModel("tiny", device="cuda", compute_type="float16")
    # 2s of silence is enough to drive a real encode() forward pass through
    # cuBLAS/cuDNN -- that call, not the constructor above, is what actually
    # fails if the DLLs aren't visible on PATH.
    silence = np.zeros(16000 * 2, dtype=np.float32)
    segments, info = m.transcribe(silence)
    list(segments)  # force the (lazy) generator to run
    print("GPU_OK")
except Exception as exc:
    print("GPU_FAIL:", repr(exc))
    sys.exit(1)
'@ | Set-Content -Path $verifyScript -Encoding UTF8

    $env:PATH = "$gpuPathExtra;$env:PATH"
    $verifyOutput = & $VenvPy $verifyScript 2>&1
    $verifyExit = $LASTEXITCODE
    Remove-Item $verifyScript -Force -ErrorAction SilentlyContinue

    Write-Host "    --- verification output ---" -ForegroundColor DarkGray
    $verifyOutput | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host "    ----------------------------" -ForegroundColor DarkGray

    if ($verifyExit -eq 0 -and ($verifyOutput -match "GPU_OK")) {
        $gpuVerified = $true
        Write-Ok "GPU acceleration VERIFIED working (ran a real transcription on cuda)."
    } else {
        Write-Warn2 "GPU acceleration did NOT verify. The tool still works fine on CPU"
        Write-Warn2 "(select Device=cpu, or leave 'auto' -- it will fall back automatically)."
        Write-Warn2 "Known-working Windows alternative if you need GPU speed:"
        Write-Warn2 "  https://github.com/Purfview/whisper-standalone-win"
    }
} else {
    Write-Ok "Skipping GPU setup (pass -WithGpu to attempt it). CPU mode works out of the box;"
    Write-Ok "expect roughly real-time-or-slower transcription on CPU with the 'base' model,"
    Write-Ok "and noticeably slower with 'medium'/'large-v3'."
}

# ---------------------------------------------------------------------------
# 5. Regenerate a portable launcher (the committed one is machine-specific)
# ---------------------------------------------------------------------------
Write-Step "Writing launch.bat"

$launchBat = Join-Path $ToolDir "launch.bat"
$pathPrefix = ""
if ($gpuVerified -and $gpuPathExtra) {
    $pathPrefix = "set PATH=$gpuPathExtra;%PATH%`r`n"
}
$launchContent = "@echo off`r`n$pathPrefix" +
    "`"$VenvPyw`" `"%~dp0app.py`" %*`r`n"
Set-Content -Path $launchBat -Value $launchContent -Encoding ASCII
Write-Ok "Launcher written to: $launchBat"

Write-Step "Done"
Write-Host "Both parts of the deployment are now set up." -ForegroundColor Green
Write-Host "Run tools\audio_processor\launch.bat to open the Audio Processor GUI." -ForegroundColor Green
if ($WithGpu) {
    if ($gpuVerified) {
        Write-Host "GPU acceleration is set up and verified working; the Device dropdown's 'auto'" -ForegroundColor Green
        Write-Host "default will detect and use it automatically (via ctranslate2.get_cuda_device_count())." -ForegroundColor Green
    } else {
        Write-Host "GPU acceleration could not be verified; use Device=cpu in the GUI." -ForegroundColor Yellow
    }
}
Write-Host ""
Write-Host "Note: the first transcription of each model size (tiny/base/small/medium/large-v3)" -ForegroundColor DarkGray
Write-Host "downloads model weights from Hugging Face and needs internet access; later runs use" -ForegroundColor DarkGray
Write-Host "the cached copy under %USERPROFILE%\.cache\huggingface." -ForegroundColor DarkGray
