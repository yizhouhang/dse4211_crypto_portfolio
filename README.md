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
Shared report runner used by every stage to export daily evaluation tables, trade logs, and plots.

`core/evaluation.py`
Shared evaluation functions used to compute metrics, trade logs, and plots.

`Adaptive/evaluation_function.py`
Compatibility wrapper kept for older notebooks that still import `evaluation_function` from the `Adaptive/` folder.

`scripts/generate_mean_reversion_tables.py`
Builds the RSI and CUSUM post-trade tables from the raw BTC/ETH data.

`scripts/run_strategy_workflow.py`
Runs the full workflow in order from standalone through EVT.

`Adaptive/rolling_combined.ipynb`
Builds the rolling combined trend-strength and volatility-filter strategy, then exports BTC/ETH diagnostics for why ETH benefits more from this rule:
- `outputs/strategy_reports/04_rolling_window/rolling_combined_diagnostics.csv`
- `outputs/strategy_reports/04_rolling_window/rolling_combined_runs.csv`
- `outputs/strategy_reports/04_rolling_window/rolling_combined_lag_events.csv`

`datasets/`
Source data and post-trade tables used by the workflow.

`outputs/strategy_reports/`
Ordered outputs for every stage. Stage folders contain:
- `data/`
- `plots/`

## Run

From the repository root:

```bash
python3 scripts/run_strategy_workflow.py
```

## Archive

Anything not part of the active workflow is kept under `archive/`.
