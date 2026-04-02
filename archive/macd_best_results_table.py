import numpy as np
import pandas as pd

btc_raw = pd.read_csv("BTC_full_data.csv")
eth_raw = pd.read_csv("ETH_full_data.csv")

# ----------------------------
# 1) prepare df
# ----------------------------
def prepare_price_data(df, date_col="Date"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    return out

btc = prepare_price_data(btc_raw)
eth = prepare_price_data(eth_raw)

import numpy as np
import pandas as pd

# ----------------------------
# 2) MACD indicator
# ----------------------------
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


# ----------------------------
# 3) Position logic
# ----------------------------
def make_position(macd, signal, mode="long_flat", buffer=0.0):
    spread = macd - signal

    if mode == "long_flat":
        pos = np.where(spread > buffer, 1, 0)

    elif mode == "long_short":
        pos = np.where(spread > buffer, 1, -1)

    elif mode == "long_short_flat":
        pos = np.where(
            spread > buffer, 1,
            np.where(spread < -buffer, -1, 0)
        )

    else:
        raise ValueError("mode must be 'long_flat', 'long_short', or 'long_short_flat'")

    return pd.Series(pos, index=macd.index)


# ----------------------------
# 4) Trade label helper
# ----------------------------
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


# ----------------------------
# 5) Generate post-trade table only
# ----------------------------
def generate_macd_post_trade_table(
    data,
    a=12,
    b=26,
    c=9,
    mode="long_flat",
    buffer=0.0,
    price_col="Close"
):
    out = add_macd(data, a=a, b=b, c=c, price_col=price_col).copy()

    # log return
    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["log_return"] = out["log_return"].fillna(0.0)

    # raw signal-based position
    raw_pos = make_position(out["macd"], out["signal"], mode=mode, buffer=buffer)

    # yesterday's decision becomes today's held position
    out["position"] = raw_pos.shift(1).fillna(0).astype(int)

    # previous held position
    prev_position = out["position"].shift(1).fillna(0).astype(int)

    # numeric trade = change in position
    # long_flat:
    #  0 -> 1  => +1
    #  1 -> 0  => -1
    #  same    => 0
    out["trade"] = (out["position"] - prev_position).astype(int)

    # readable trade label
    out["trade_action"] = [
        get_trade_action(prev, curr)
        for prev, curr in zip(prev_position, out["position"])
    ]

    # handle date column
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
    final_table = temp[["date", "price", "log_return", "trade", "trade_action", "position"]].copy()

    return final_table

eth_macd_table = generate_macd_post_trade_table(
    data=eth,
    a=18,
    b=50,
    c=13,
    mode="long_flat",
    buffer=0.0,
    price_col="Close"
)

print(eth_macd_table.head(15))

btc_macd_table = generate_macd_post_trade_table(
    data=btc,
    a=16,
    b=20,
    c=15,
    mode="long_flat",
    buffer=0.0,
    price_col="Close"
)

print(btc_macd_table.head(15))

btc_macd_table.to_csv("btc_macd_16_20_15_post_trade_table.csv", index=False)
eth_macd_table.to_csv("eth_macd_18_50_13_post_trade_table.csv", index=False)