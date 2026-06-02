"""Alpha Vantage provider — REST API for historical OHLCV.

Free tier: 25 calls/day (then 5 calls/min). Covers US + EU + most global markets.
Get a key: https://www.alphavantage.co/support/#api-key (instant, no credit card)

Docs: https://www.alphavantage.co/documentation/

Notes
-----
* The free tier is rate-limited per minute AND per day — DO NOT call this in
  a tight loop. The resilience layer will space out retries, but for bulk
  historical fetches prefer Finnhub or Twelve Data.
* Intraday intervals: 1min, 5min, 15min, 30min, 60min.
* Daily: TIME_SERIES_DAILY (covers full daily history).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx
import pandas as pd

from app.data.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.alphavantage.co/query"
INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min",
}


class AlphaVantageProvider(BaseProvider):
    name = "alpha_vantage"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def coverage(self) -> dict:
        return {
            "markets": ["US", "EU", "UK", "CH", "AS"],
            "intervals": ["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
            "intraday": True,
            "has_fundamentals": True,
        }

    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None

        # Decide which AV function to call
        if interval in INTERVAL_MAP:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": INTERVAL_MAP[interval],
                "outputsize": "full" if _period_to_count(period) > 100 else "compact",
                "apikey": self._api_key,
            }
            ts_key = f"Time Series ({INTERVAL_MAP[interval]})"
        else:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self._api_key,
            }
            ts_key = "Time Series (Daily)"

        try:
            with httpx.Client(timeout=30) as client:
                r = client.get(ENDPOINT, params=params)
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Alpha Vantage network error: {exc}", kind="timeout") from exc

        # Rate limit / error responses come as {"Note": ...} or {"Information": ...}
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            if "call frequency" in msg.lower() or "premium" in msg.lower():
                raise ProviderError(f"Alpha Vantage rate limit: {msg}", kind="429")
            raise ProviderError(f"Alpha Vantage: {msg}", kind="other")

        if "Error Message" in data:
            raise ProviderError(f"Alpha Vantage: {data['Error Message']}", kind="404")

        ts = data.get(ts_key)
        if not ts:
            raise ProviderError(f"Alpha Vantage no data for {symbol}", kind="404")

        # AV returns dict {timestamp: {1. open, 2. high, 3. low, 4. close, 5. volume}}
        rows = []
        for ts_str, vals in ts.items():
            rows.append({
                "timestamp": pd.Timestamp(ts_str),
                "open": float(vals.get("1. open", 0)),
                "high": float(vals.get("2. high", 0)),
                "low": float(vals.get("3. low", 0)),
                "close": float(vals.get("4. close", 0)),
                "volume": float(vals.get("5. volume", 0)),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return self._clean(self._normalize_columns(df))

    def fetch_quote(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(ENDPOINT, params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self._api_key,
                })
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Alpha Vantage network error: {exc}", kind="timeout") from exc

        if "Note" in data or "Information" in data:
            raise ProviderError("Alpha Vantage rate limit", kind="429")
        quote = data.get("Global Quote") or {}
        price_str = quote.get("05. price")
        if not price_str:
            raise ProviderError(f"Alpha Vantage no quote for {symbol}", kind="404")
        return float(price_str)

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 429:
            raise ProviderError("Alpha Vantage rate limit", kind="429", status_code=429)
        if r.status_code == 403:
            raise ProviderError("Alpha Vantage auth failed", kind="other", status_code=403)
        if r.status_code == 404:
            raise ProviderError("Alpha Vantage 404", kind="404", status_code=404)
        if r.status_code >= 500:
            raise ProviderError(f"Alpha Vantage {r.status_code}", kind="5xx", status_code=r.status_code)
        r.raise_for_status()


def _period_to_count(period: str) -> int:
    """Approximate number of bars for a period (used to pick compact vs full)."""
    p = period.strip().lower()
    m = re.match(r"(\d+)([dwmoy])", p)
    if not m:
        return 100
    val, unit = int(m.group(1)), m.group(2)
    if unit == "d": return val
    if unit == "w": return val * 5
    if unit in ("mo", "m"): return val * 22
    if unit == "y": return val * 252
    return 100
