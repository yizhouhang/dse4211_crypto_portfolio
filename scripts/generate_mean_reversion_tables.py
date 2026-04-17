from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"


def load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    computed_log_return = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    if "Log_Return" in df.columns:
        df["Log_Return"] = df["Log_Return"].fillna(computed_log_return)
    else:
        df["Log_Return"] = computed_log_return

    return df


def build_trade_columns(position: pd.Series) -> tuple[pd.Series, pd.Series]:
    trade = position.diff().fillna(position).astype(int)

    def classify(change: int) -> str:
        if change > 0:
            return "buy"
        if change < 0:
            return "sell"
        return "hold"

    trade_action = trade.apply(classify)
    return trade, trade_action


def make_post_trade_table(df: pd.DataFrame, position_col: str) -> pd.DataFrame:
    position = df[position_col].fillna(0).astype(int)
    trade, trade_action = build_trade_columns(position)

    return pd.DataFrame(
        {
            "date": df["Date"],
            "price": df["Close"],
            "log_return": df["Log_Return"].fillna(0.0),
            "trade": trade,
            "trade_action": trade_action,
            "position": position,
        }
    )


def apply_rsi_mean_reversion(
    df: pd.DataFrame,
    window: int = 14,
    lower: int = 30,
    upper: int = 70,
) -> pd.DataFrame:
    data = df.copy()
    data["delta"] = data["Close"].diff()
    data["gain"] = data["delta"].clip(lower=0)
    data["loss"] = -data["delta"].clip(upper=0)
    data["avg_gain"] = data["gain"].rolling(window=window, min_periods=window).mean()
    data["avg_loss"] = data["loss"].rolling(window=window, min_periods=window).mean()

    rs = data["avg_gain"] / (data["avg_loss"] + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["signal"] = np.select(
        [data["RSI"] < lower, data["RSI"] > upper],
        [1, -1],
        default=0,
    )
    data["position"] = data["signal"].shift(1).fillna(0).astype(int)
    return data


def apply_cusum_mean_reversion(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.05,
) -> pd.DataFrame:
    data = df.copy()
    data["mean_ret"] = data["Log_Return"].rolling(window).mean()
    data["demeaned"] = data["Log_Return"] - data["mean_ret"]
    data["CUSUM"] = data["demeaned"].rolling(window).sum()

    position = []
    current_pos = 0

    for value in data["CUSUM"]:
        if pd.isna(value):
            position.append(0)
            continue

        if current_pos == 0:
            if value > threshold:
                current_pos = -1
            elif value < -threshold:
                current_pos = 1
        elif current_pos == 1 and value >= 0:
            current_pos = 0
        elif current_pos == -1 and value <= 0:
            current_pos = 0

        position.append(current_pos)

    data["position"] = pd.Series(position, index=data.index).astype(int)
    return data


def export_strategy_tables(asset_slug: str, market_path: Path) -> list[Path]:
    market_df = load_market_data(market_path)

    strategy_tables = {
        DATASETS_DIR / f"{asset_slug}_rsi_post_trade_table.csv": make_post_trade_table(
            apply_rsi_mean_reversion(market_df),
            "position",
        ),
        DATASETS_DIR / f"{asset_slug}_cusum_post_trade_table.csv": make_post_trade_table(
            apply_cusum_mean_reversion(market_df),
            "position",
        ),
    }

    written = []
    for path, table in strategy_tables.items():
        table.to_csv(path, index=False)
        written.append(path)

    return written


def main():
    generated = []
    generated.extend(export_strategy_tables("btc", DATASETS_DIR / "BTC_full_data.csv"))
    generated.extend(export_strategy_tables("eth", DATASETS_DIR / "ETH_full_data.csv"))

    print(f"Generated {len(generated)} mean-reversion post-trade tables.")
    for path in generated:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
