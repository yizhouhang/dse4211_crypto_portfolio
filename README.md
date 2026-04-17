# DSE4211 Crypto Portfolio

## Relevant Files

`strategy_workflow/01_standalone/run_reports.py`
Runs the 5 standalone strategy groups across BTC and ETH:
- MACD
- Turtle
- RSI
- Bollinger Bands
- CUSUM mean reversion

`strategy_workflow/02_filters/run_reports.py`
Runs the volatility-filter and trend-strength strategies.

`strategy_workflow/03_combined/run_reports.py`
Runs the combined filter strategies.

`strategy_workflow/04_rolling_window/run_reports.py`
Runs the rolling-window strategy reports.

`strategy_workflow/05_evt/run_reports.py`
Runs the EVT-based trend-strength risk filter reports.

`strategy_workflow/common.py`
Shared report runner used by every stage.

`core/evaluation.py`
Shared evaluation functions used to compute metrics, trade logs, and plots.

`Adaptive/evaluation_function.py`
Compatibility wrapper kept for older notebooks that still import `evaluation_function` from the `Adaptive/` folder.

`scripts/generate_mean_reversion_tables.py`
Builds the RSI and CUSUM post-trade tables from the raw BTC/ETH data.

`scripts/run_strategy_workflow.py`
Runs the full workflow in order from standalone through EVT.

`datasets/`
Source data and post-trade tables used by the workflow.

`outputs/strategy_reports/`
Ordered outputs for every stage. Each stage contains:
- `metrics_summary.csv`
- `data/`
- `plots/`

## Run

From the repository root:

```bash
python3 scripts/run_strategy_workflow.py
```

## Archive

Anything not part of the active workflow is kept under `archive/`.
