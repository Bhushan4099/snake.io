@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First launch: setting up Python environment and dependencies...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -c "import torch, pygame, numpy" >nul 2>&1
if errorlevel 1 (
  echo Installing missing Snake AI Lab dependencies...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 exit /b 1
)

echo.
echo === Snake AI Lab: train, evaluate, then launch ===
echo Training DQN for 500 episodes on CPU...
".venv\Scripts\python.exe" "%~dp0train.py" --episodes 500 --seed 42
if errorlevel 1 (
  echo Training failed. Check the error above.
  pause
  exit /b 1
)

echo.
echo Evaluating DQN and random baseline over 100 episodes...
".venv\Scripts\python.exe" "%~dp0evaluate.py" --compare --episodes 100
if errorlevel 1 (
  echo Evaluation failed. Check the error above.
  pause
  exit /b 1
)

echo.
echo Opening Snake AI Lab. Select AI Auto Play from the menu.
".venv\Scripts\python.exe" "%~dp0run.py"
if errorlevel 1 pause
endlocal
