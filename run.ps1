<#
Snake AI Lab - one-command setup and run for Windows (PowerShell 5.1+).

  .\run.bat -Quick      # setup + tests + quick smoke pipeline -> results_quick\  (a few minutes)
  .\run.bat             # setup + tests + full pipeline (5 seeds x 2 variants x 500 episodes) -> results\
  .\run.bat -UI         # setup + open the Pygame menu
  .\run.bat -Quick -SkipTests

run.bat calls this script with -ExecutionPolicy Bypass, so no policy change is needed.
Steps: find Python 3.12, create/reuse .venv312, install requirements (only if missing),
run pytest, then run the pipeline (or the UI).
#>
param(
    [switch]$Quick,
    [switch]$UI,
    [switch]$SkipTests,
    [int]$Seeds = 0,
    [int]$Workers = 2
)
# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr line of a native program (pip warnings) into an error.
# Failures are caught explicitly through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

$Venv = Join-Path $PSScriptRoot ".venv312"
$Py = Join-Path $Venv "Scripts\python.exe"

function Invoke-Checked([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe $($ArgList -join ' ')" }
}

# 1. Python 3.12 virtual environment
if (-not (Test-Path $Py)) {
    Write-Host "== Creating virtual environment .venv312 (Python 3.12) =="
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $Venv
        $made = ($LASTEXITCODE -eq 0) -and (Test-Path $Py)
    }
    if (-not $made -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
        if ($ver -eq "3.12") {
            & python -m venv $Venv
            $made = ($LASTEXITCODE -eq 0) -and (Test-Path $Py)
        }
    }
    if (-not $made) {
        throw "Python 3.12 not found. Install it from https://www.python.org/downloads/ (3.12.x, 64-bit), then re-run. (Python 3.14 has no pygame wheel.)"
    }
}
$ver = (& $Py -c "import sys; print('%d.%d' % sys.version_info[:2])")
Write-Host "Using $Py (Python $ver)"
if ($ver -ne "3.12") { Write-Warning "Expected Python 3.12 in .venv312, found $ver" }

# 2. Dependencies (skipped when already importable)
& $Py -c "import torch, numpy, pygame, matplotlib, pytest" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "== Installing dependencies (first run only; torch is a large download) =="
    Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
    # CPU-only torch first (smaller); fall back to PyPI if the CPU index is unreachable.
    & $Py -m pip install --timeout 120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Invoke-Checked $Py @("-m", "pip", "install", "--timeout", "120", "--retries", "5", "torch") }
    Invoke-Checked $Py @("-m", "pip", "install", "--timeout", "120", "--retries", "5", "-r", "requirements.txt")
} else {
    Write-Host "Dependencies already installed."
}

# 3. UI mode
if ($UI) {
    Write-Host "== Starting the Pygame menu =="
    Invoke-Checked $Py @("main.py")
    exit 0
}

# 4. Tests
if (-not $SkipTests) {
    Write-Host "== Running tests =="
    Invoke-Checked $Py @("-m", "pytest", "-q")
}

# 5. Pipeline
$runArgs = @("run_all.py", "--workers", "$Workers")
if ($Quick) {
    Write-Host "== Quick pipeline -> results_quick\ (results\ is not touched) =="
    $runArgs += "--quick"
    if ($Seeds -gt 0) { $runArgs += @("--seeds", "$Seeds") }
} else {
    if ($Seeds -le 0) { $Seeds = 5 }   # the reported results use 5 seeds
    Write-Host "== Full pipeline ($Seeds seeds x 2 variants x 500 episodes) -> results\ (overwrites it; ~30+ min on 2 cores) =="
    $runArgs += @("--seeds", "$Seeds")
}
$t0 = Get-Date
Invoke-Checked $Py $runArgs
Write-Host ("Pipeline finished in {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)
