from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, run_stage_reports


def main():
    specs = [
        {"name": "BTC Combined Filter", "slug": "btc_combined", "csv_path": ROOT / "datasets" / "btc_combined_strategy.csv", "category": "combined"},
        {"name": "ETH Combined Filter", "slug": "eth_combined", "csv_path": ROOT / "datasets" / "eth_combined_strategy.csv", "category": "combined"},
    ]
    run_stage_reports("03_combined", specs)


if __name__ == "__main__":
    main()
