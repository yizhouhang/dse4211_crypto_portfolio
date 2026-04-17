from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, run_stage_reports


def main():
    specs = [
        {"name": "BTC Rolling Trend Strength", "slug": "btc_rolling_trend_strength", "csv_path": ROOT / "datasets" / "btc_rolling_trend_strength.csv", "category": "rolling_window"},
        {"name": "ETH Rolling Trend Strength", "slug": "eth_rolling_trend_strength", "csv_path": ROOT / "datasets" / "eth_rolling_trend_strength.csv", "category": "rolling_window"},
        {"name": "BTC Rolling Volatility Filter", "slug": "btc_rolling_vol", "csv_path": ROOT / "datasets" / "btc_rolling_vol.csv", "category": "rolling_window"},
        {"name": "ETH Rolling Volatility Filter", "slug": "eth_rolling_vol", "csv_path": ROOT / "datasets" / "eth_rolling_vol.csv", "category": "rolling_window"},
    ]
    run_stage_reports("04_rolling_window", specs)


if __name__ == "__main__":
    main()
