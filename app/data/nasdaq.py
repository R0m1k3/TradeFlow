"""
TradeFlow — NASDAQ Stock List
Fetches and caches the full list of NASDAQ-listed tickers.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Fallback list of major NASDAQ stocks by market cap (used if fetch fails)
TOP_NASDAQ = [
    # Mega cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "GOOGL", "META", "TSLA",
    # Large cap
    "AVGO", "COST", "NFLX", "AMD", "CRM", "ADBE", "PEP", "CSCO",
    "INTC", "CMCSA", "TXN", "QCOM", "SBUX", "AMGN", "VRTX",
    "BKNG", "ISRG", "REGN", "FISV", "GILD", "ADP", "CSX", "MU",
    "LRCX", "PANW", "KLAC", "SNPS", "MCHP", "CDW", "MAR", "MRVL",
    "ORLY", "FTNT", "MNST", "DLTR", "CPRT", "CTAS", "FAST",
    "IDXX", "BIIB", "VRSK", "ILMN", "CEG", "WDAY", "XEL", "EXC",
    "WBA", "MDLZ", "KDP", "ROST", "CHTR", "PYPL",
    # Tech growth
    "ABNB", "ZS", "DDOG", "CRWD", "NET", "AXON", "MELI", "TEAM",
    "MDB", "PSTG", "HUBS", "OKTA", "SNOW", "PLTR", "RBLX", "U",
    "COIN", "SHOP", "MSTR", "SIRI", "LCID", "RIVN", "NIO",
    # Semis & hardware
    "ON", "MRAM", "SWKS", "QRVO", "SMCI", "MPWR", "POWI",
    # Biotech & pharma
    "MRNA", "BNTX", "VTRS", "TECH", "ALKS", "DXCM", "ICLR",
    # Fintech & payments
    "SQ", "PATH", "AFRM", "UPST", "HOOD",
    # Energy & industrials
    "ENPH", "SEDG", "FSLR", "GEHC",
    # Consumer
    "LULU", "DLTH", "BURL", "TGT",
    # Crypto-adjacent
    "MSTR", "RIOT", "CLSK", "MARA",
    # Other notable NASDAQ
    "PCAR", "FSLR", "VRSK", "EXC", "XEL",
]

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "nasdaq_tickers.csv"


@lru_cache(maxsize=1)
def get_nasdaq_tickers() -> list[str]:
    """
    Return the full list of NASDAQ tickers.
    Tries to fetch from NASDAQ, falls back to TOP_NASDAQ.
    """
    # Try cache file first
    if CACHE_PATH.exists():
        try:
            df = pd.read_csv(CACHE_PATH)
            if "Symbol" in df.columns and len(df) > 100:
                tickers = df["Symbol"].dropna().str.strip().str.upper().tolist()
                logger.info("Loaded %d NASDAQ tickers from cache", len(tickers))
                return tickers
        except Exception:
            pass

    # Try fetching from NASDAQ FTP
    try:
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=NASDAQ"
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("data", {}).get("rows", [])
            if rows:
                tickers = [r["symbol"] for r in rows if r.get("symbol")]
                logger.info("Fetched %d NASDAQ tickers from API", len(tickers))
                # Cache to file
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"Symbol": tickers}).to_csv(CACHE_PATH, index=False)
                return tickers
    except Exception as exc:
        logger.warning("NASDAQ API fetch failed: %s", exc)

    # Try yfinance approach
    try:
        import yfinance as yf
        # Get NASDAQ 100 tickers as a reasonable alternative
        nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(nasdaq_url)
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].dropna().str.strip().str.upper().tolist()
                logger.info("Fetched %d NASDAQ-100 tickers from Wikipedia", len(tickers))
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"Symbol": tickers}).to_csv(CACHE_PATH, index=False)
                return tickers
    except Exception as exc:
        logger.warning("Wikipedia NASDAQ fetch failed: %s", exc)

    logger.info("Using fallback NASDAQ list (%d tickers)", len(TOP_NASDAQ))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in TOP_NASDAQ:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def search_tickers(query: str, limit: int = 20) -> list[str]:
    """Search NASDAQ tickers by prefix. Returns up to `limit` matches."""
    query = query.strip().upper()
    if not query:
        return TOP_NASDAQ[:limit]
    all_tickers = get_nasdaq_tickers()
    matches = [t for t in all_tickers if t.startswith(query)]
    if not matches:
        matches = [t for t in all_tickers if query in t]
    return matches[:limit]