# ======================================================================================
# STRATEGY: MACD & BOLLINGER PULLBACK HYBRID
# --------------------------------------------------------------------------------------
# PURPOSE: 
#   Combines Trend following (MACD) with Mean-Reversion (Bollinger Bands) to identify 
#   high-probability entry points during market pullbacks.
#
# LOGIC:
#   1. MACD Filter: Ensures the overall trend is bullish (MACD > Signal).
#   2. Bollinger Pullback: Entries occur when price dips toward or below the 
#      lower Bollinger Band (oversold) while the MACD trend remains intact.
#   3. Risk Management: Includes a hard Stop-Loss and Z-score based exit logic.
#
# INPUT: raw eth and btc price data (CSV)
# OUTPUT: Post-trade table with signals, trades, and position tracking.
# ======================================================================================

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_price_data(df, date_col="Date"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def add_macd(data, a=12, b=26, c=9, price_col="Close"):
    out = data.copy()

    if a >= b:
        raise ValueError("Need a < b for MACD.")

    out["ema_fast"] = out[price_col].ewm(span=a, adjust=False).mean()
    out["ema_slow"] = out[price_col].ewm(span=b, adjust=False).mean()
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["signal"] = out["macd"].ewm(span=c, adjust=False).mean()
    out["hist"] = out["macd"] - out["signal"]

    return out


def add_bollinger_features(data, window=30, num_std=1.0, price_col="Close"):
    out = data.copy()

    out["rolling_mean"] = out[price_col].rolling(window).mean()
    out["rolling_std"] = out[price_col].rolling(window).std()
    out["upper_band"] = out["rolling_mean"] + num_std * out["rolling_std"]
    out["lower_band"] = out["rolling_mean"] - num_std * out["rolling_std"]

    rolling_std_nonzero = out["rolling_std"].replace(0, np.nan)
    out["bb_z"] = (out[price_col] - out["rolling_mean"]) / rolling_std_nonzero

    return out


def get_trade_action(prev_pos, curr_pos):
    if curr_pos == prev_pos:
        return "hold"

    if prev_pos == 0 and curr_pos == 1:
        return "buy"
    if prev_pos == 1 and curr_pos == 0:
        return "sell"

    if prev_pos == 0 and curr_pos == -1:
        return "sell_short"
    if prev_pos == -1 and curr_pos == 0:
        return "buy_to_cover"

    if prev_pos == -1 and curr_pos == 1:
        return "reverse_to_long"
    if prev_pos == 1 and curr_pos == -1:
        return "reverse_to_short"

    return "hold"


def _validate_post_trade_table(table):
    required_cols = ["date", "price", "log_return", "trade", "trade_action", "position"]
    missing = [col for col in required_cols if col not in table.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")

    if not table["date"].is_monotonic_increasing:
        raise ValueError("Output table must be sorted by date.")

    if not set(table["position"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Hybrid strategy should only emit long-flat positions (0 or 1).")

    if len(table) > 0:
        first_row = table.iloc[0]
        if float(first_row["log_return"]) != 0.0:
            raise ValueError("First row must have log_return = 0.0.")
        if int(first_row["position"]) != 0:
            raise ValueError("First row must have position = 0.")
        if int(first_row["trade"]) != 0:
            raise ValueError("First row must have trade = 0.")


def generate_macd_bollinger_pullback_post_trade_table(
    data,
    a,
    b,
    c,
    bb_window=20,
    bb_num_std=1.0,
    entry_z=-0.3,
    exit_z=0.0,
    exit_mode="macd_only",
    stop_loss=-0.08,
    price_col="Close",
):
    out = prepare_price_data(data).copy()
    out = add_macd(out, a=a, b=b, c=c, price_col=price_col)
    out = add_bollinger_features(out, window=bb_window, num_std=bb_num_std, price_col=price_col)

    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["log_return"] = out["log_return"].fillna(0.0)

    raw_positions = []
    current_position = 0
    entry_price = np.nan

    for row in out.itertuples():
        spread = row.macd - row.signal
        bb_z = row.bb_z
        price = getattr(row, price_col)

        if pd.isna(spread) or pd.isna(bb_z):
            current_position = 0
            entry_price = np.nan
        elif current_position == 0:
            if spread > 0 and bb_z <= entry_z:
                current_position = 1
                entry_price = price
        else:
            stop_loss_hit = (
                stop_loss is not None
                and pd.notna(entry_price)
                and entry_price != 0
                and ((price / entry_price) - 1) <= stop_loss
            )

            if exit_mode == "macd_only":
                exit_now = (spread <= 0) or stop_loss_hit
            elif exit_mode == "mean_or_macd":
                exit_now = (bb_z >= exit_z) or (spread <= 0) or stop_loss_hit
            else:
                raise ValueError("exit_mode must be 'macd_only' or 'mean_or_macd'")

            if exit_now:
                current_position = 0
                entry_price = np.nan

        raw_positions.append(current_position)

    out["raw_position"] = pd.Series(raw_positions, index=out.index).astype(int)

    # Use yesterday's desired position as today's held position to avoid lookahead bias.
    out["position"] = out["raw_position"].shift(1).fillna(0).astype(int)

    prev_position = out["position"].shift(1).fillna(0).astype(int)
    out["trade"] = (out["position"] - prev_position).astype(int)
    out["trade_action"] = [
        get_trade_action(prev, curr)
        for prev, curr in zip(prev_position, out["position"])
    ]

    temp = out.copy()

    if isinstance(temp.index, pd.DatetimeIndex):
        temp = temp.reset_index()
        date_col = temp.columns[0]
    elif "Date" in temp.columns:
        date_col = "Date"
    elif "date" in temp.columns:
        date_col = "date"
    else:
        temp = temp.reset_index()
        date_col = temp.columns[0]

    temp = temp.rename(columns={date_col: "date", price_col: "price"})

    final_table = temp[
        ["date", "price", "log_return", "trade", "trade_action", "position"]
    ].copy()
    _validate_post_trade_table(final_table)

    return final_table


def _resolve_input_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "datasets" / filename,
        script_dir / filename,
        script_dir / "datasets" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


def _print_basic_checks(asset_name, table):
    print(f"\n--- {asset_name} Basic Checks ---")
    print(f"rows: {len(table)}")
    print(f"date range: {table['date'].min()} -> {table['date'].max()}")
    print(f"positions: {sorted(table['position'].unique().tolist())}")
    print(f"trades: {int(table['trade'].abs().sum())}")
    print(table.head(5).to_string(index=False))


def _try_evaluate_with_evaluation_function(asset_name, table):
    evaluator_path = Path(__file__).resolve().parent / "evaluation_function.py"

    try:
        source = evaluator_path.read_text(encoding="utf-8")
        filtered_lines = []
        for line in source.splitlines():
            if line.strip().startswith("eth_macd = pd.read_csv("):
                continue
            filtered_lines.append(line)

        module_globals = {"__name__": "evaluation_function_sanitized"}
        exec(compile("\n".join(filtered_lines), str(evaluator_path), "exec"), module_globals)
        evaluate_strategy_full = module_globals["evaluate_strategy_full"]
    except Exception as exc:
        print(
            f"\nSkipping evaluation_function import for {asset_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    try:
        _, _, summary = evaluate_strategy_full(post_trade_df=table, fee=0.0, rf_annual=0.03)
    except Exception as exc:
        print(
            f"\nevaluation_function failed for {asset_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print(f"\n--- {asset_name} Evaluator Summary ---")
    for metric, value in summary.items():
        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")


def run_default_exports():
    btc_path = _resolve_input_path("BTC_full_data.csv")
    eth_path = _resolve_input_path("ETH_full_data.csv")

    btc_raw = pd.read_csv(btc_path)
    eth_raw = pd.read_csv(eth_path)

    btc_table = generate_macd_bollinger_pullback_post_trade_table(
        data=btc_raw,
        a=16,
        b=20,
        c=15,
        bb_window=20,
        bb_num_std=1.0,
        entry_z=-0.3,
        exit_z=0.0,
        exit_mode="macd_only",
        stop_loss=-0.08,
        price_col="Close",
    )
    eth_table = generate_macd_bollinger_pullback_post_trade_table(
        data=eth_raw,
        a=18,
        b=50,
        c=13,
        bb_window=20,
        bb_num_std=1.0,
        entry_z=-0.3,
        exit_z=0.0,
        exit_mode="macd_only",
        stop_loss=-0.08,
        price_col="Close",
    )

    btc_output = Path("btc_macd_bollinger_pullback_post_trade_table.csv")
    eth_output = Path("eth_macd_bollinger_pullback_post_trade_table.csv")

    btc_table.to_csv(btc_output, index=False)
    eth_table.to_csv(eth_output, index=False)

    print(f"Exported BTC hybrid table -> {btc_output}")
    print(f"Exported ETH hybrid table -> {eth_output}")

    _print_basic_checks("BTC", btc_table)
    _print_basic_checks("ETH", eth_table)

    _try_evaluate_with_evaluation_function("BTC", btc_table)
    _try_evaluate_with_evaluation_function("ETH", eth_table)


if __name__ == "__main__":
    run_default_exports()
