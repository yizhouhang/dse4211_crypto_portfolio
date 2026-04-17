from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, run_stage_reports


def main():
    specs = [
        {"name": "BTC Volatility Filter", "slug": "btc_vol_filter", "csv_path": ROOT / "datasets" / "btc_vol_filter_post_trade.csv", "category": "volatility_filter"},
        {"name": "ETH Volatility Filter", "slug": "eth_vol_filter", "csv_path": ROOT / "datasets" / "eth_vol_filter_post_trade.csv", "category": "volatility_filter"},
        {"name": "BTC Trend Strength", "slug": "btc_trend_strength", "csv_path": ROOT / "datasets" / "btc_trend_strength.csv", "category": "trend_strength"},
        {"name": "ETH Trend Strength", "slug": "eth_trend_strength", "csv_path": ROOT / "datasets" / "eth_trend_strength.csv", "category": "trend_strength"},
    ]
    run_stage_reports("02_filters", specs)


if __name__ == "__main__":
    main()
