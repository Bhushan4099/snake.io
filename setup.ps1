$ErrorActionPreference = "Stop"

Write-Host "=== Snake AI Lab: one-time setup ===" -ForegroundColor Cyan

function Test-PythonCommand {
    param(
        [Parameter(Mandatory = $true)][string]$CommandPath,
        [string[]]$PrefixArgs = @()
    )

    $versionCheck = "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    & $CommandPath @PrefixArgs -c $versionCheck 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE. Fix the error above and rerun .\setup.ps1."
    }
}

$venvDirectory = Join-Path (Get-Location) ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

# Clean up an incomplete virtual environment left by a failed earlier setup.
if ((Test-Path $venvDirectory) -and -not (Test-Path $venvPython)) {
    Write-Host "Removing incomplete .venv from the previous setup attempt..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvDirectory
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Looking for an installed Python 3.11 or newer..." -ForegroundColor Yellow

    $selectedCommand = $null
    $selectedPrefix = @()

    # Prefer python.exe on PATH. This avoids a common Windows py.exe issue:
    # py -3.11 emits stderr if 3.11 is absent; with ErrorActionPreference=Stop,
    # that can abort discovery before trying an installed 3.12 runtime.
    foreach ($name in @("python", "python3")) {
        $candidate = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $candidate -and (Test-PythonCommand -CommandPath $candidate.Source)) {
            $selectedCommand = $candidate.Source
            $selectedPrefix = @()
            break
        }
    }

    # Fallback: enumerate registered runtimes and invoke their executable paths
    # directly, avoiding py -3.xx version selectors.
    if ($null -eq $selectedCommand) {
        $pyCommand = Get-Command "py" -ErrorAction SilentlyContinue
        if ($null -ne $pyCommand) {
            $runtimeList = @(& $pyCommand.Source -0p 2>$null)
            foreach ($line in $runtimeList) {
                if ($line -match '(?i)([A-Z]:\\[^\r\n]+\python\.exe)\s*$') {
                    $runtimePath = $Matches[1]
                    if ((Test-Path $runtimePath -PathType Leaf) -and (Test-PythonCommand -CommandPath $runtimePath)) {
                        $selectedCommand = $runtimePath
                        $selectedPrefix = @()
                        break
                    }
                }
            }
        }
    }

    if ($null -eq $selectedCommand) {
        Write-Host "No Python 3.11+ runtime was found." -ForegroundColor Red
        Write-Host "Check installed versions with: py -0p" -ForegroundColor Yellow
        Write-Host "Also try: python --version" -ForegroundColor Yellow
        throw "Install Python 3.11 or newer, enable its PATH option, reopen VS Code, and rerun .\setup.ps1."
    }

    Write-Host "Using Python runtime:" -ForegroundColor Green
    & $selectedCommand @selectedPrefix --version
    if ($LASTEXITCODE -ne 0) { throw "Could not read the selected Python version." }

    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    Invoke-Checked -Description "Virtual environment creation" -Command {
        & $selectedCommand @selectedPrefix -m venv .venv
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation did not produce .venv\Scripts\python.exe. Check your Python installation and permissions."
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
Invoke-Checked -Description "pip upgrade" -Command {
    & $venvPython -m pip install --upgrade pip
}

Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Cyan
Invoke-Checked -Description "CPU PyTorch installation" -Command {
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Installing Snake AI Lab, Pygame and test dependencies..." -ForegroundColor Cyan
Invoke-Checked -Description "Project dependency installation" -Command {
    & $venvPython -m pip install -e ".[dev,game]"
}

Write-Host "Running tests..." -ForegroundColor Cyan
Invoke-Checked -Description "Test suite" -Command {
    & $venvPython -m pytest
}

Write-Host "Setup complete. Start the application with: .\.venv\Scripts\python.exe run.py" -ForegroundColor Green
