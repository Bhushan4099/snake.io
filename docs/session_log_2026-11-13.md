---
date: 2026-11-13
project: snake-ai-lab
tags: [snake-ai, dqn, pygame, session-log]
---

# Session log — 2026-11-13 (day before review)

Review date: 14 Nov 2026.

## 1. Windows setup (first native-Windows run)

**Problem 1:** `python -m pytest` → `No module named pytest`. The first venv (`.venv`) was on **Python 3.14**; torch installed but the second `pip install` line never ran.

**Problem 2:** installing into the 3.14 venv failed: `ERROR: Failed to build 'pygame'`. pygame 2.6.1 has no prebuilt wheel for 3.14, so pip tried to compile it from source (`No module named 'setuptools._distutils.msvccompiler'`).

**Problem 3:** a pip `ReadTimeoutError` from files.pythonhosted.org (slow network).

**Fix:** new venv on Python 3.12 (the project's target), retried with a longer timeout.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --timeout 120 --retries 5 numpy pygame matplotlib pytest
python -m pytest -q      # 35 passed
python main.py
```

Installed: Python 3.12.10, torch 2.15.0+cpu, pygame 2.6.1.

**Lesson:** use Python 3.12 for this project; check `python --version` inside the venv before installing.

## 2. Speed meter in Manual Play

- New reusable `Slider` class in `snakelab/ui.py` (drag with the mouse, or `+` / `-` keys).
- Manual Play: 2–30 steps/s, default 8 (about the old fixed 120 ms per step).
- AI Auto Play uses the same class: 1–60 steps/s, default 10, +/- step 2.
- Manual Play panel now also shows the step count.

## 3. Minimal pixel theme (black / blue / red)

- Palette: black background, dark-blue grid, **blue** text and snake body, light-blue head, **red** for titles, food, the selected menu item, the agent's chosen action and the slider knob, dim blue for hints.
- Square blocks everywhere (no rounded corners); menu items are outlined boxes with a `>` marker.
- New `snakelab/pixelfont.py`: a hand-made 5x7 bitmap font (A–Z, 0–9, punctuation), drawn as solid squares at scale 2 (body) and 3 (titles). Text is shown in uppercase.
  - A first attempt (default font rendered small without anti-aliasing, scaled 2x) was unreadable ("SCORE" looked like "SCDRE"), so it was replaced with the bitmap font.

## 4. Verification

- `pytest`: 35 passed on Windows, Python 3.12.
- All screens rendered headless and inspected; README screenshots in `docs/screenshots/` regenerated.
- Evaluate screen (DQN seed 0, final checkpoint, 100 greedy games): mean 21.85, std 8.39, max 45; random mean 0.14.
- Played by the user on the real Windows display: done.

## Files changed

- `snakelab/ui.py`: palette, `Slider`, Manual Play slider, restyled screens
- `snakelab/pixelfont.py`: new bitmap font
- `README.md`: Manual Play row mentions the slider; feature row mentions the theme
- `docs/screenshots/*.png`: regenerated

## Still open

- Viva prep: walk through `docs/viva_questions.md` and the demo order.
- The 19 features were never checked against hand-built boards.
