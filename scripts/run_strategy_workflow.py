from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPTS = [
    ROOT / "strategy_workflow" / "01_standalone" / "run_reports.py",
    ROOT / "strategy_workflow" / "02_filters" / "run_reports.py",
    ROOT / "strategy_workflow" / "03_combined" / "run_reports.py",
    ROOT / "strategy_workflow" / "04_rolling_window" / "run_reports.py",
    ROOT / "strategy_workflow" / "05_evt" / "run_reports.py",
]


def main():
    for stage_script in STAGE_SCRIPTS:
        subprocess.run([sys.executable, str(stage_script)], check=True, cwd=ROOT)

    print("Completed strategy workflow in order:")
    for stage_script in STAGE_SCRIPTS:
        print(f"- {stage_script.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
