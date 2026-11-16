#!/usr/bin/env bash
# Snake AI Lab - one-command setup and run for Linux / WSL2 / macOS.
#   ./run_all.sh --quick        setup + tests + quick pipeline -> results_quick/ (results/ untouched)
#   ./run_all.sh                setup + tests + full pipeline (5 seeds) -> results/ (overwrites it)
#   ./run_all.sh --ui           setup + Pygame menu (needs a display; WSL2 needs WSLg)
#   ./run_all.sh --quick --skip-tests
# Uses python3.12 if present (override with PYTHON=/path/to/python), venv in .venv (override with VENV=dir).
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
UI=0; SKIP_TESTS=0; QUICK=0; EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --ui) UI=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --quick) QUICK=1; EXTRA+=("$arg") ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then PY="$VENV/Scripts/python.exe"
else
  BASE="${PYTHON:-$(command -v python3.12 || command -v python3 || command -v python)}"
  echo "== Creating $VENV with $BASE ($("$BASE" --version)) =="
  "$BASE" -m venv "$VENV"
  if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"; else PY="$VENV/Scripts/python.exe"; fi
fi
echo "Using $PY ($("$PY" --version))"

if ! "$PY" -c "import torch, numpy, pygame, matplotlib, pytest" 2>/dev/null; then
  echo "== Installing dependencies (first run only) =="
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install --timeout 120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install --timeout 120 --retries 5 torch
  "$PY" -m pip install --timeout 120 --retries 5 -r requirements.txt
fi

if [ "$UI" = 1 ]; then exec "$PY" main.py; fi
if [ "$SKIP_TESTS" = 0 ]; then "$PY" -m pytest -q; fi

if [ "$QUICK" = 0 ] && ! printf '%s\n' "${EXTRA[@]+"${EXTRA[@]}"}" | grep -q -- '--seeds'; then
  EXTRA+=(--seeds 5)   # the reported results use 5 seeds
fi
"$PY" run_all.py "${EXTRA[@]}"
