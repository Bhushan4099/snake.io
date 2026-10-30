"""Single entry point: python run.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    missing: list[str] = []
    for module, package in (("numpy", "numpy"), ("torch", "torch"), ("pygame", "pygame")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Install them once, then rerun:")
        if "torch" in missing:
            print("  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
        remaining = [item for item in missing if item != "torch"]
        if remaining:
            print("  python -m pip install " + " ".join(remaining))
        return 1

    from snake_ai_lab.game.app import main as launch_game
    launch_game()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
