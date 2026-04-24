import logging
import sys
import os

# Ensure the app module can be found
sys.path.insert(0, os.path.abspath("."))

from app.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.DEBUG)

def main():
    print("Testing fetcher...")
    df = fetch_ohlcv("AAPL", interval="1d", period="1mo", use_cache=False)
    if df is not None:
        print(f"Success! Fetched {len(df)} rows.")
        print(df.head())
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
