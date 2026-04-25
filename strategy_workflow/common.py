from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import pandas as pd

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
EVAL_MODULE_PATH = ROOT / "core" / "evaluation.py"
REPORTS_ROOT = ROOT / "outputs" / "strategy_reports"


def load_evaluation_module():
    spec = importlib.util.spec_from_file_location("evaluation_function", EVAL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def export_strategy_report(
    module,
    *,
    stage_slug: str,
    strategy_name: str,
    strategy_slug: str,
    table_df: pd.DataFrame,
    position_col: str = "position",
):
    stage_dir = REPORTS_ROOT / stage_slug
    data_dir = stage_dir / "data"
    plots_dir = stage_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    daily_df, trade_log, _summary = module.evaluate_strategy_full(
        post_trade_df=table_df,
        position_col=position_col,
        fee=0.0,
        rf_annual=0.03,
        trading_days=365,
    )

    daily_path = data_dir / f"{strategy_slug}_daily_eval.csv"
    trade_log_path = data_dir / f"{strategy_slug}_trade_log.csv"
    daily_df.to_csv(daily_path, index=False)
    trade_log.to_csv(trade_log_path, index=False)

    module.plot_strategy_vs_buy_hold(
        daily_df,
        title=f"{strategy_name} vs Buy-and-Hold",
        output_path=plots_dir / f"{strategy_slug}_vs_buy_hold.png",
        show=False,
    )
    module.plot_equity_curve(
        daily_df,
        title=f"{strategy_name} Equity Curve",
        output_path=plots_dir / f"{strategy_slug}_equity_curve.png",
        show=False,
    )
    module.plot_drawdown(
        daily_df,
        title=f"{strategy_name} Drawdown",
        output_path=plots_dir / f"{strategy_slug}_drawdown.png",
        show=False,
    )


def run_stage_reports(stage_slug: str, strategy_specs: list[dict]):
    module = load_evaluation_module()

    for spec in strategy_specs:
        if "table_df" in spec:
            table_df = module.normalize_strategy_table(spec["table_df"])
        else:
            table_df = module.normalize_strategy_table(pd.read_csv(spec["csv_path"]))

        export_strategy_report(
            module,
            stage_slug=stage_slug,
            strategy_name=spec["name"],
            strategy_slug=spec["slug"],
            table_df=table_df,
            position_col=spec.get("position_col", "position"),
        )

    print(f"{stage_slug}: generated {len(strategy_specs)} strategy reports")
