"""Finnhub provider — REST API for historical OHLCV.

Free tier: 60 calls/min, covers US + EU + Asia.
Get a key: https://finnhub.io/register (instant, no credit card)

Docs: https://finnhub.io/docs/api/stock-candles
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from app.data.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

ENDPOINT = "https://finnhub.io/api/v1"
INTERVAL_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "1d": "D", "1w": "W", "1M": "M",
}


class FinnhubProvider(BaseProvider):
    name = "finnhub"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def coverage(self) -> dict:
        return {
            "markets": ["US", "EU", "UK", "CH", "AS"],
            "intervals": ["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
            "intraday": True,
            "has_fundamentals": True,
        }

    def _resolve(self, symbol: str) -> str:
        """Finnhub uses exchange prefixes: 'AAPL' (US), 'MC.PA' (Euronext),
        'ROG.SW' (Swiss), 'VOD.L' (London). The symbols we already use map
        directly — no transformation needed."""
        return symbol

    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None
        if interval not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")

        # Convert period like "3mo" → seconds-from-now
        now = int(time.time())
        from_ts = now - _period_to_seconds(period)

        try:
            with httpx.Client(timeout=20) as client:
                r = client.get(
                    f"{ENDPOINT}/stock/candle",
                    params={
                        "symbol": self._resolve(symbol),
                        "resolution": INTERVAL_MAP[interval],
                        "from": from_ts,
                        "to": now,
                    },
                    headers={"X-Finnhub-Token": self._api_key},
                )
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Finnhub network error: {exc}", kind="timeout") from exc

        if data.get("s") != "ok":
            # Common: "s": "no_data" — ticker not found or no data in range
            raise ProviderError(f"Finnhub no_data for {symbol}", kind="404")

        timestamps = data.get("t") or []
        if not timestamps:
            return None

        df = pd.DataFrame({
            "open": data.get("o", []),
            "high": data.get("h", []),
            "low": data.get("l", []),
            "close": data.get("c", []),
            "volume": data.get("v", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        return self._clean(self._normalize_columns(df))

    def fetch_quote(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(
                    f"{ENDPOINT}/quote",
                    params={"symbol": self._resolve(symbol)},
                    headers={"X-Finnhub-Token": self._api_key},
                )
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Finnhub network error: {exc}", kind="timeout") from exc

        price = data.get("c")  # current price
        if not price or price == 0:
            raise ProviderError(f"Finnhub no quote for {symbol}", kind="404")
        return float(price)

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 429:
            raise ProviderError("Finnhub rate limit", kind="429", status_code=429)
        if r.status_code == 403:
            raise ProviderError("Finnhub auth failed (check API key)", kind="other", status_code=403)
        if r.status_code == 404:
            raise ProviderError("Finnhub 404", kind="404", status_code=404)
        if r.status_code >= 500:
            raise ProviderError(f"Finnhub {r.status_code}", kind="5xx", status_code=r.status_code)
        r.raise_for_status()


def _period_to_seconds(period: str) -> int:
    """Convert yfinance-style period to seconds."""
    p = period.strip().lower()
    if p == "max":
        return 60 * 60 * 24 * 365 * 30  # 30 years
    import re
    m = re.match(r"(\d+)([dwmoy])", p)
    if not m:
        return 60 * 60 * 24 * 90  # default 3mo
    val, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return val * 86400
    if unit == "w":
        return val * 7 * 86400
    if unit in ("mo", "m"):
        return val * 30 * 86400
    if unit == "y":
        return val * 365 * 86400
    return 60 * 60 * 24 * 90
