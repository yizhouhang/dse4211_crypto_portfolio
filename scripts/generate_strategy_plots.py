from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import pandas as pd

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
EVAL_MODULE_PATH = ROOT / "core" / "evaluation.py"
OUTPUT_DIR = ROOT / "outputs" / "plots" / "strategy_pnl"
MEAN_REVERSION_GENERATOR = ROOT / "scripts" / "generate_mean_reversion_tables.py"

STRATEGY_SPECS = [
    {
        "name": "BTC MACD",
        "slug": "btc_macd",
        "csv_path": ROOT / "datasets" / "btc_macd_16_20_15_post_trade_table.csv",
        "kind": "standalone",
        "family": "trend_following",
    },
    {
        "name": "ETH MACD",
        "slug": "eth_macd",
        "csv_path": ROOT / "datasets" / "eth_macd_18_50_13_post_trade_table.csv",
        "kind": "standalone",
        "family": "trend_following",
    },
    {
        "name": "BTC Bollinger",
        "slug": "btc_bollinger",
        "csv_path": ROOT / "datasets" / "btc_bollinger_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "ETH Bollinger",
        "slug": "eth_bollinger",
        "csv_path": ROOT / "datasets" / "eth_bollinger_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "BTC RSI",
        "slug": "btc_rsi",
        "csv_path": ROOT / "datasets" / "btc_rsi_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "ETH RSI",
        "slug": "eth_rsi",
        "csv_path": ROOT / "datasets" / "eth_rsi_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "BTC CUSUM Mean Reversion",
        "slug": "btc_cusum",
        "csv_path": ROOT / "datasets" / "btc_cusum_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "ETH CUSUM Mean Reversion",
        "slug": "eth_cusum",
        "csv_path": ROOT / "datasets" / "eth_cusum_post_trade_table.csv",
        "kind": "standalone",
        "family": "mean_reversion",
    },
    {
        "name": "BTC Turtle",
        "slug": "btc_turtle",
        "csv_path": ROOT / "datasets" / "btc_turtle_table.csv",
        "kind": "standalone",
        "family": "trend_following",
    },
    {
        "name": "ETH Turtle",
        "slug": "eth_turtle",
        "csv_path": ROOT / "datasets" / "eth_turtle_table.csv",
        "kind": "standalone",
        "family": "trend_following",
    },
    {
        "name": "BTC Trend Strength Adaptive",
        "slug": "btc_trend_strength",
        "csv_path": ROOT / "datasets" / "btc_trend_strength.csv",
        "kind": "adaptive",
    },
    {
        "name": "ETH Trend Strength Adaptive",
        "slug": "eth_trend_strength",
        "csv_path": ROOT / "datasets" / "eth_trend_strength.csv",
        "kind": "adaptive",
    },
    {
        "name": "BTC Volatility Filter Adaptive",
        "slug": "btc_vol_filter",
        "csv_path": ROOT / "datasets" / "btc_vol_filter_post_trade.csv",
        "kind": "adaptive",
    },
    {
        "name": "ETH Volatility Filter Adaptive",
        "slug": "eth_vol_filter",
        "csv_path": ROOT / "datasets" / "eth_vol_filter_post_trade.csv",
        "kind": "adaptive",
    },
    {
        "name": "BTC Combined Adaptive",
        "slug": "btc_combined",
        "csv_path": ROOT / "datasets" / "btc_combined_strategy.csv",
        "kind": "adaptive",
    },
    {
        "name": "ETH Combined Adaptive",
        "slug": "eth_combined",
        "csv_path": ROOT / "datasets" / "eth_combined_strategy.csv",
        "kind": "adaptive",
    },
]


def load_evaluation_module():
    spec = importlib.util.spec_from_file_location("evaluation_function", EVAL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_required_strategy_tables():
    subprocess.run(
        [sys.executable, str(MEAN_REVERSION_GENERATOR)],
        check=True,
        cwd=ROOT,
    )


def plot_one_strategy(module, spec: dict) -> dict:
    raw_df = pd.read_csv(spec["csv_path"])
    strategy_df = module.normalize_strategy_table(raw_df)

    daily_df, _, summary = module.evaluate_strategy_full(
        post_trade_df=strategy_df,
        fee=0.0,
        rf_annual=0.03,
        trading_days=365,
    )

    output_dir = OUTPUT_DIR / spec["kind"]
    output_path = output_dir / f"{spec['slug']}_pnl_curve.png"

    module.plot_strategy_vs_buy_hold(
        daily_df,
        title=f"{spec['name']} vs Buy-and-Hold",
        output_path=output_path,
        show=False,
    )

    return {
        "strategy": spec["name"],
        "category": spec["kind"],
        "family": spec.get("family", ""),
        "output_path": str(output_path.relative_to(ROOT)),
        "cumulative_pnl": summary["cumulative_pnl"],
        "sharpe_ratio_rf_3pct": summary["sharpe_ratio_rf_3pct"],
        "max_drawdown": summary["max_drawdown"],
    }


def main():
    ensure_required_strategy_tables()
    module = load_evaluation_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in STRATEGY_SPECS:
        rows.append(plot_one_strategy(module, spec))

    summary_df = pd.DataFrame(rows).sort_values(["category", "strategy"]).reset_index(drop=True)
    summary_path = OUTPUT_DIR / "strategy_plot_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Generated {len(summary_df)} PnL plots.")
    print(f"Summary saved to {summary_path.relative_to(ROOT)}")
    for row in rows:
        print(f"- {row['strategy']}: {row['output_path']}")


if __name__ == "__main__":
    main()
