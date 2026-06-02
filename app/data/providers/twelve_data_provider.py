"""Twelve Data provider — REST API for historical OHLCV.

Free tier: 800 calls/day, 8 calls/min. Covers 100+ exchanges globally.
Get a key: https://twelvedata.com/pricing (free signup, no credit card)

Docs: https://twelvedata.com/docs

Notes
-----
* Twelve Data is the best free tier for European coverage: Swiss (.SWX),
  German (.XETRA), French (.EPA), Amsterdam (.AMS), London (.LSE).
* Symbols can be passed as "ROG.SWX" or "ROG:SIX" (interchangeable).
* Supports WebSocket for real-time if needed.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
import pandas as pd

from app.data.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.twelvedata.com"
INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "1d": "1day", "1w": "1week", "1M": "1month",
}


class TwelveDataProvider(BaseProvider):
    name = "twelve_data"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def coverage(self) -> dict:
        return {
            "markets": ["US", "EU", "UK", "CH", "DE", "FR", "NL", "AS", "CRYPTO", "FX"],
            "intervals": ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"],
            "intraday": True,
            "has_fundamentals": True,
        }

    def _resolve(self, symbol: str) -> str:
        """Twelve Data prefers `EXCHANGE:SYMBOL` for clarity. We map common
        Yahoo suffixes to their Twelve Data exchange codes:
            .PA → EPA   (Euronext Paris)
            .AS → AMS   (Euronext Amsterdam)
            .SW → SWX   (Swiss Exchange)
            .L → LSE    (London)
            .DE → XETRA
            .MI → MIL   (Borsa Italiana)
        """
        s = symbol.upper()
        if s.endswith(".PA"):
            return f"EPA:{s[:-3]}"
        if s.endswith(".AS"):
            return f"AMS:{s[:-3]}"
        if s.endswith(".SW"):
            return f"SWX:{s[:-3]}"
        if s.endswith(".L"):
            return f"LSE:{s[:-2]}"
        if s.endswith(".DE"):
            return f"XETRA:{s[:-3]}"
        if s.endswith(".MI"):
            return f"MIL:{s[:-3]}"
        return s

    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None
        if interval not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")

        try:
            with httpx.Client(timeout=20) as client:
                r = client.get(
                    f"{ENDPOINT}/time_series",
                    params={
                        "symbol": self._resolve(symbol),
                        "interval": INTERVAL_MAP[interval],
                        "outputsize": 5000,
                        "apikey": self._api_key,
                    },
                )
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Twelve Data network error: {exc}", kind="timeout") from exc

        if data.get("status") == "error":
            msg = data.get("message", "")
            if "not found" in msg.lower() or "symbol" in msg.lower():
                raise ProviderError(f"Twelve Data: {msg}", kind="404")
            raise ProviderError(f"Twelve Data: {msg}", kind="other")

        values = data.get("values") or []
        if not values:
            raise ProviderError(f"Twelve Data no data for {symbol}", kind="404")

        rows = []
        for v in values:
            rows.append({
                "timestamp": pd.Timestamp(v["datetime"]),
                "open": float(v.get("open", 0)),
                "high": float(v.get("high", 0)),
                "low": float(v.get("low", 0)),
                "close": float(v.get("close", 0)),
                "volume": float(v.get("volume", 0)),
            })
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return self._clean(self._normalize_columns(df))

    def fetch_quote(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(
                    f"{ENDPOINT}/quote",
                    params={"symbol": self._resolve(symbol), "apikey": self._api_key},
                )
                self._check(r)
                data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Twelve Data network error: {exc}", kind="timeout") from exc

        if data.get("status") == "error":
            raise ProviderError(f"Twelve Data: {data.get('message', 'unknown')}", kind="404")
        price = data.get("close")
        if not price:
            raise ProviderError(f"Twelve Data no quote for {symbol}", kind="404")
        return float(price)

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 429:
            raise ProviderError("Twelve Data rate limit", kind="429", status_code=429)
        if r.status_code == 401:
            raise ProviderError("Twelve Data auth failed", kind="other", status_code=401)
        if r.status_code == 404:
            raise ProviderError("Twelve Data 404", kind="404", status_code=404)
        if r.status_code >= 500:
            raise ProviderError(f"Twelve Data {r.status_code}", kind="5xx", status_code=r.status_code)
        r.raise_for_status()
