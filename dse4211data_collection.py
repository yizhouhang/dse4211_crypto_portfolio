import yfinance as yf
import pandas as pd
import numpy as np

# Define tickers and date range
tickers = ["BTC-USD", "ETH-USD"]
start_date = "2018-01-01"
end_date = "2025-12-31"

def download_crypto_full(ticker, start, end):
    print(f"Downloading all columns for {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, multi_level_index=False)

    if df.empty:
        return None

    # 1. Clean column names (removes spaces/formatting issues)
    df.columns = [str(col).strip() for col in df.columns]

    # 2. Calculate Log Return using 'Adj Close'
    # Use .ffill() to handle any missing price days before calculating
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # 3. Reorder columns to your preferred layout
    ordered_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Log_Return']
    
    # Only keep columns that actually exist in the download
    final_cols = [c for c in ordered_cols if c in df.columns]
    
    return df[final_cols]

# Download BTC and ETH
btc_data = download_crypto_full("BTC-USD", start_date, end_date)
eth_data = download_crypto_full("ETH-USD", start_date, end_date)

# Save and Preview
if btc_data is not None:
    btc_data.to_csv("BTC_full_data.csv")
    print("\n--- BTC Full Data (Last 5 days) ---")
    print(btc_data.tail())

if eth_data is not None:
    eth_data.to_csv("ETH_full_data.csv")
    print("\n--- ETH Full Data (Last 5 days) ---")
    print(eth_data.tail())