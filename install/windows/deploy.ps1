<#
.SYNOPSIS
  Deploys the SegmentHumanBody Slicer module (annotation-process-recorder branch)
  for local use on a machine with no standalone Python installation.

.DESCRIPTION
  3D Slicer ships its own bundled Python interpreter (PythonSlicer.exe), so the
  module itself needs no system Python. This script:
    1. Locates an existing 3D Slicer install (does NOT install Slicer itself).
    2. Fetches/updates the module source (git if available, else a zip download
       via .NET so it also works on machines without git).
    3. Installs the module's two optional runtime dependencies (sounddevice for
       audio recording, scikit-image for the SLIC/Felzenszwalb superpixel
       models) into Slicer's bundled Python.
    4. Writes a one-click launcher batch file that starts Slicer with the
       module folder on its module search path.

.PARAMETER SlicerExe
  Full path to Slicer.exe. If omitted, the script searches common locations.

.PARAMETER RepoUrl
  Git remote to clone/pull from. Defaults to the fork this branch lives on.

.PARAMETER Branch
  Branch to deploy. Defaults to annotation-process-recorder.

.PARAMETER RepoDir
  Where to place/update the checkout. Defaults to .\SlicerSegmentHumanBody
  next to this script -- unless this script is itself already sitting inside
  a checkout of the repo (e.g. you got it via `install/windows/deploy.ps1`
  from a clone or a GitHub zip download), in which case that checkout is
  used directly and nothing is downloaded.

.PARAMETER WithSuperpixelModels
  Also install scikit-image (needed only for the SPX_SLIC2D / SPX_Felzenszwalb2D
  model choices; the default Identity model and the recorder work without it).

.EXAMPLE
  .\deploy.ps1
  .\deploy.ps1 -WithSuperpixelModels
  .\deploy.ps1 -SlicerExe "C:\Program Files\Slicer 5.12.3\Slicer.exe"
#>

[CmdletBinding()]
param(
    [string]$SlicerExe = "",
    [string]$RepoUrl = "https://github.com/yzluka/SlicerSegmentHumanBody.git",
    [string]$Branch = "annotation-process-recorder",
    [string]$RepoDir = "",
    [switch]$WithSuperpixelModels
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# Walks up from $startDir looking for a folder that looks like the repo root
# (has SegmentHumanBody/SegmentHumanBody.py). Returns $null if not found
# within a few levels. Not a fixed relative offset -- this script may live at
# install/windows/deploy.ps1 (repo root two levels up) today, or move again
# later (e.g. once install/mac, install/linux exist).
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

$UseInPlaceCheckout = $false
if ([string]::IsNullOrWhiteSpace($RepoDir)) {
    if ($DetectedRepoRoot) {
        $RepoDir = $DetectedRepoRoot
        $UseInPlaceCheckout = $true
    } else {
        $RepoDir = Join-Path $ScriptDir "SlicerSegmentHumanBody"
    }
}

# ---------------------------------------------------------------------------
# 1. Locate 3D Slicer (bundled Python lives inside it - nothing else to install)
# ---------------------------------------------------------------------------
Write-Step "Looking for a 3D Slicer installation"

if ([string]::IsNullOrWhiteSpace($SlicerExe)) {
    $candidates = @()
    $candidates += Get-ChildItem -Path $ScriptDir -Filter "Slicer.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue
    # Also check the repo root: when this script lives at
    # repo_root\install\windows\, a Slicer install placed alongside the repo
    # (repo_root\Slicer\...) is nowhere under $ScriptDir.
    if ($DetectedRepoRoot) {
        $candidates += Get-ChildItem -Path $DetectedRepoRoot -Filter "Slicer.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue
    }
    foreach ($root in @("$env:ProgramFiles", "$env:ProgramFiles(x86)", "$env:LocalAppData\slicer.org", "$env:LocalAppData\NA-MIC")) {
        if (Test-Path $root) {
            $candidates += Get-ChildItem -Path $root -Filter "Slicer.exe" -Recurse -Depth 3 -ErrorAction SilentlyContinue
        }
    }
    $found = $candidates | Select-Object -First 1
    if ($found) { $SlicerExe = $found.FullName }
}

if ([string]::IsNullOrWhiteSpace($SlicerExe) -or -not (Test-Path $SlicerExe)) {
    Write-Warn2 "No 3D Slicer install found automatically."
    Write-Host ""
    Write-Host "This machine needs 3D Slicer itself (it bundles its own Python, so no" -ForegroundColor Yellow
    Write-Host "separate Python install is required). Download and run the installer from:" -ForegroundColor Yellow
    Write-Host "  https://download.slicer.org/" -ForegroundColor Yellow
    Write-Host "Then re-run this script, or pass -SlicerExe `"<path to Slicer.exe>`"." -ForegroundColor Yellow
    exit 1
}
$SlicerDir = Split-Path -Parent $SlicerExe
$PythonSlicer = Join-Path $SlicerDir "bin\PythonSlicer.exe"
if (-not (Test-Path $PythonSlicer)) {
    # Some layouts put Slicer.exe itself in the "bin"-equivalent root.
    $PythonSlicer = Join-Path $SlicerDir "PythonSlicer.exe"
}
Write-Ok "Slicer.exe:      $SlicerExe"
Write-Ok "PythonSlicer.exe: $PythonSlicer"
if (-not (Test-Path $PythonSlicer)) {
    throw "Found Slicer.exe but not PythonSlicer.exe next to/under it - unexpected install layout."
}

# ---------------------------------------------------------------------------
# 2. Fetch/update the module source
# ---------------------------------------------------------------------------
Write-Step "Fetching module source ($RepoUrl @ $Branch)"

if ($UseInPlaceCheckout) {
    Write-Ok "This script is already sitting inside a checkout of the repo -- using"
    Write-Ok "$RepoDir directly. Not touching its git state (no fetch/reset), in case"
    Write-Ok "it's a working copy with local changes -- update it yourself (git pull,"
    Write-Ok "or re-download) if you want the latest."
    $git = $null
} else {

$git = Get-Command git -ErrorAction SilentlyContinue

if (Test-Path (Join-Path $RepoDir ".git")) {
    if ($git) {
        Write-Ok "Existing checkout found, pulling latest..."
        git -C $RepoDir fetch --quiet origin $Branch
        if ($LASTEXITCODE -ne 0) { throw "git fetch failed (exit $LASTEXITCODE)" }
        git -C $RepoDir checkout --quiet $Branch
        if ($LASTEXITCODE -ne 0) { throw "git checkout failed (exit $LASTEXITCODE)" }
        git -C $RepoDir reset --quiet --hard "origin/$Branch"
        if ($LASTEXITCODE -ne 0) { throw "git reset failed (exit $LASTEXITCODE)" }
        Write-Ok "Updated $RepoDir to latest $Branch"
    } else {
        Write-Warn2 "git not found but a git checkout already exists at $RepoDir - leaving it as-is."
    }
} elseif ($git) {
    git clone --branch $Branch --single-branch $RepoUrl $RepoDir
    Write-Ok "Cloned into $RepoDir"
} else {
    # No git available: download the branch as a zip and extract it - pure
    # PowerShell/.NET, no external tools required.
    Write-Warn2 "git not found - downloading a zip snapshot instead."
    $owner, $name = ($RepoUrl -replace '\.git$', '') -split '/' | Select-Object -Last 2
    $zipUrl = "https://github.com/$owner/$name/archive/refs/heads/$Branch.zip"
    $tmpZip = Join-Path $env:TEMP "SlicerSegmentHumanBody_$Branch.zip"
    $tmpExtract = Join-Path $env:TEMP "SlicerSegmentHumanBody_extract"

    Invoke-WebRequest -Uri $zipUrl -OutFile $tmpZip -UseBasicParsing
    if (Test-Path $tmpExtract) { Remove-Item $tmpExtract -Recurse -Force }
    Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force

    $extractedRoot = Get-ChildItem $tmpExtract | Select-Object -First 1
    if (Test-Path $RepoDir) { Remove-Item $RepoDir -Recurse -Force }
    Move-Item $extractedRoot.FullName $RepoDir

    Remove-Item $tmpZip -Force
    Remove-Item $tmpExtract -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok "Downloaded and extracted into $RepoDir"
}

} # end of else ($UseInPlaceCheckout)

$ModuleDir = Join-Path $RepoDir "SegmentHumanBody"
if (-not (Test-Path (Join-Path $ModuleDir "SegmentHumanBody.py"))) {
    throw "SegmentHumanBody.py not found under $ModuleDir - checkout looks incomplete."
}

# ---------------------------------------------------------------------------
# 3. Install optional runtime dependencies into Slicer's bundled Python
# ---------------------------------------------------------------------------
Write-Step "Installing optional Python dependencies into Slicer's bundled Python"

# sounddevice: needed only for the audio-recording checkbox. If missing, the
# module runs fine and audio recording silently no-ops (per module docs).
& $PythonSlicer -m pip install --quiet --disable-pip-version-check sounddevice
if ($LASTEXITCODE -eq 0) {
    Write-Ok "sounddevice installed (enables audio recording)."
} else {
    Write-Warn2 "sounddevice install failed - audio recording will be unavailable, everything else still works."
}

if ($WithSuperpixelModels) {
    & $PythonSlicer -m pip install --quiet --disable-pip-version-check scikit-image
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "scikit-image installed (enables SPX_SLIC2D / SPX_Felzenszwalb2D models)."
    } else {
        Write-Warn2 "scikit-image install failed - those two model choices will be unavailable."
    }
} else {
    Write-Ok "Skipping scikit-image (pass -WithSuperpixelModels to install it for the SLIC/Felzenszwalb models)."
}

# ---------------------------------------------------------------------------
# 4. Write a one-click launcher
# ---------------------------------------------------------------------------
Write-Step "Writing launcher"

# In standalone mode $ScriptDir IS the top-level project folder -- put it
# there. In in-place mode $ScriptDir is repo_root\install\windows\, too deep
# to be convenient, so put it at the repo root instead, where it's easy to find.
$LauncherDir = if ($UseInPlaceCheckout) { $DetectedRepoRoot } else { $ScriptDir }
$LauncherPath = Join-Path $LauncherDir "Run_SegmentHumanBody.bat"
$launcherContent = "@echo off`r`n`"$SlicerExe`" --additional-module-path `"$ModuleDir`"`r`n"
Set-Content -Path $LauncherPath -Value $launcherContent -Encoding ASCII

Write-Ok "Launcher written to: $LauncherPath"

Write-Step "Done"
Write-Host "Double-click Run_SegmentHumanBody.bat (or re-run it) to start Slicer with the" -ForegroundColor Green
Write-Host "module registered. In Slicer, use the module search box (top toolbar) and" -ForegroundColor Green
Write-Host "type 'SegmentHumanBody' (category: Segmentation) to open it." -ForegroundColor Green
Write-Host ""
Write-Host "Next: run deploy_audio_processor.ps1 -- the audio transcription tool is the" -ForegroundColor Green
Write-Host "other half of this deployment (turns recorded WAV+JSON into text reports)." -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File .\deploy_audio_processor.ps1" -ForegroundColor Green
