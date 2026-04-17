from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, run_stage_reports


def ensure_mean_reversion_tables():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_mean_reversion_tables.py")],
        check=True,
        cwd=ROOT,
    )


def main():
    ensure_mean_reversion_tables()

    specs = [
        {"name": "BTC MACD", "slug": "btc_macd", "csv_path": ROOT / "datasets" / "btc_macd_16_20_15_post_trade_table.csv", "category": "trend_following"},
        {"name": "ETH MACD", "slug": "eth_macd", "csv_path": ROOT / "datasets" / "eth_macd_18_50_13_post_trade_table.csv", "category": "trend_following"},
        {"name": "BTC Turtle", "slug": "btc_turtle", "csv_path": ROOT / "datasets" / "btc_turtle_table.csv", "category": "trend_following"},
        {"name": "ETH Turtle", "slug": "eth_turtle", "csv_path": ROOT / "datasets" / "eth_turtle_table.csv", "category": "trend_following"},
        {"name": "BTC RSI", "slug": "btc_rsi", "csv_path": ROOT / "datasets" / "btc_rsi_post_trade_table.csv", "category": "mean_reversion"},
        {"name": "ETH RSI", "slug": "eth_rsi", "csv_path": ROOT / "datasets" / "eth_rsi_post_trade_table.csv", "category": "mean_reversion"},
        {"name": "BTC Bollinger", "slug": "btc_bollinger", "csv_path": ROOT / "datasets" / "btc_bollinger_post_trade_table.csv", "category": "mean_reversion"},
        {"name": "ETH Bollinger", "slug": "eth_bollinger", "csv_path": ROOT / "datasets" / "eth_bollinger_post_trade_table.csv", "category": "mean_reversion"},
        {"name": "BTC CUSUM Mean Reversion", "slug": "btc_cusum", "csv_path": ROOT / "datasets" / "btc_cusum_post_trade_table.csv", "category": "mean_reversion"},
        {"name": "ETH CUSUM Mean Reversion", "slug": "eth_cusum", "csv_path": ROOT / "datasets" / "eth_cusum_post_trade_table.csv", "category": "mean_reversion"},
    ]
    run_stage_reports("01_standalone", specs)


if __name__ == "__main__":
    main()
