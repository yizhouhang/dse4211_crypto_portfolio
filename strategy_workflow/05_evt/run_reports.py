from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, run_stage_reports


def calculate_rolling_evt_es(returns: pd.Series, window: int = 250, confidence: float = 0.95, threshold_pct: float = 0.90):
    es_series = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]
        losses = -window_returns[window_returns < 0]

        if len(losses) < 25:
            es_series.iloc[i] = -window_returns.quantile(1 - confidence)
            continue

        u = losses.quantile(threshold_pct)
        exceedances = losses[losses > u] - u

        try:
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            n = len(losses)
            nu = len(exceedances)
            p = 1 - confidence
            var_evt = u + (scale / shape) * (((n / nu) * p) ** (-shape) - 1)
            es_evt = (var_evt + scale - shape * u) / (1 - shape)
            es_series.iloc[i] = es_evt
        except Exception:
            es_series.iloc[i] = losses.mean()

    return es_series.ffill()


def build_evt_table(base_csv: Path) -> pd.DataFrame:
    common = __import__("common")
    module = common.load_evaluation_module()

    base_df = module.normalize_strategy_table(pd.read_csv(base_csv))
    daily_df, _, _ = module.evaluate_strategy_full(post_trade_df=base_df, fee=0.0, rf_annual=0.03, trading_days=365)
    daily_df["evt_es"] = calculate_rolling_evt_es(daily_df["strategy_ret_net"])
    risk_threshold = daily_df["evt_es"].expanding().quantile(0.90)
    daily_df["position_evt"] = np.where(daily_df["evt_es"] > risk_threshold, 0, daily_df["position"]).astype(int)

    evt_df = daily_df[["date", "price", "log_return", "position_evt"]].copy()
    evt_df = evt_df.rename(columns={"position_evt": "position"})
    evt_df["trade"] = evt_df["position"].diff().fillna(evt_df["position"]).astype(int)
    evt_df["trade_action"] = evt_df["trade"].map(lambda x: "buy" if x > 0 else ("sell" if x < 0 else "hold"))
    return evt_df


def main():
    specs = [
        {"name": "BTC Trend Strength + EVT", "slug": "btc_trend_strength_evt", "table_df": build_evt_table(ROOT / "datasets" / "btc_trend_strength.csv"), "category": "evt"},
        {"name": "ETH Trend Strength + EVT", "slug": "eth_trend_strength_evt", "table_df": build_evt_table(ROOT / "datasets" / "eth_trend_strength.csv"), "category": "evt"},
    ]
    run_stage_reports("05_evt", specs)


if __name__ == "__main__":
    main()
