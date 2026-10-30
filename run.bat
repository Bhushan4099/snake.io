@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First launch: setting up Python environment and dependencies...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 exit /b 1
) else (
  ".venv\Scripts\python.exe" -c "import torch, pygame, numpy" >nul 2>&1
  if errorlevel 1 (
    echo Installing missing Snake AI Lab dependencies...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 exit /b 1
  )
)

".venv\Scripts\python.exe" "%~dp0run.py"
if errorlevel 1 pause
endlocal
