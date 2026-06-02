"""Yahoo Finance provider — yfinance library.

Always-available fallback (no key required). Known issues:
* Rate-limited (429) under heavy load
* Some tickers (e.g. ROG.SW Roche Genussschein) return 404
* Recent API instability — use as last-resort fallback only
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import pandas as pd

from app.data.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Mirror the fetcher's interval limits
INTERVAL_PERIOD_LIMITS = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "730d", "1d": "max", "1wk": "max", "1mo": "max",
}


def _cap_period(interval: str, period: str) -> str:
    max_p = INTERVAL_PERIOD_LIMITS.get(interval)
    if max_p is None or max_p == "max":
        return period
    p = _to_days(period); m = _to_days(max_p)
    if p is not None and m is not None and p > m:
        return max_p
    return period


def _to_days(period: str) -> int | None:
    p = period.strip().lower()
    if p == "max": return None
    m = re.match(r"(\d+)([dwmoy])", p)
    if not m: return None
    val, u = int(m.group(1)), m.group(2)
    return val if u == "d" else val * 7 if u == "w" else val * 30 if u in ("mo", "m") else val * 365 if u == "y" else None


class YahooProvider(BaseProvider):
    name = "yahoo"

    def is_available(self) -> bool:
        # Always available — no key required
        return True

    def coverage(self) -> dict:
        return {
            "markets": ["US", "EU", "UK", "CH", "AS", "FX", "CRYPTO"],
            "intervals": ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
            "intraday": True,
            "has_fundamentals": False,
        }

    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderError("yfinance not installed", kind="other") from exc

        if interval not in INTERVAL_PERIOD_LIMITS:
            raise ValueError(f"Unsupported interval: {interval}")

        actual_period = _cap_period(interval, period)
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=actual_period, interval=interval, auto_adjust=True)
        except Exception as exc:
            raise ProviderError(f"yfinance error: {exc}", kind="other") from exc

        if raw is None or raw.empty:
            # Could be a 404-style "no data" — surface as 404 so the
            # negative cache treats it as a permanent failure.
            raise ProviderError(f"yfinance no data for {symbol}", kind="404")

        raw.columns = [str(c).lower() for c in raw.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        df = raw[keep].copy()
        return self._clean(df)

    def fetch_quote(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderError("yfinance not installed", kind="other") from exc
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="5m", auto_adjust=True)
            if hist is None or hist.empty:
                # Try fast_info as a fallback
                try:
                    fi = ticker.fast_info
                    last = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
                    if last:
                        return float(last)
                except Exception:
                    pass
                raise ProviderError(f"yfinance no quote for {symbol}", kind="404")
            col = "close" if "close" in hist.columns else "Close"
            return float(hist.iloc[-1][col])
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"yfinance error: {exc}", kind="other") from exc
