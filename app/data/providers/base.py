"""Abstract base class for market data providers.

Each provider (Finnhub, Alpha Vantage, Twelve Data, MT5, Yahoo, Stooq) implements
this interface. The `SourceRouter` then chains them in priority order, with the
resilience layer handling failures and backoff.

Provider contract
-----------------
* `name`             : short id ("finnhub", "alpha_vantage", …)
* `is_available()`   : True if configured (API key present, etc.)
* `fetch_ohlcv(...)` : returns a pandas DataFrame or None
* `fetch_quote(...)` : returns a float (current price) or None
* `coverage()`       : returns a dict of supported markets/intervals

Node.js porting notes
---------------------
The same interface applies. In Node:
  class BaseProvider {
    async fetchOhlcv(symbol, interval, period) { throw new Error('not impl'); }
    async fetchQuote(symbol) { throw new Error('not impl'); }
  }
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when a provider call fails (network, auth, rate limit, etc.)."""
    def __init__(self, message: str, *, kind: str = "other", status_code: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind           # "404" | "5xx" | "timeout" | "429" | "other"
        self.status_code = status_code


class BaseProvider(ABC):
    """Abstract base for all market data providers."""

    name: str = "base"

    def __init__(self) -> None:
        # Per-request API key (BYO-key override). Setter: set_request_key().
        # If empty, falls back to SettingsStore → env var.
        self._request_key: str = ""

    def set_request_key(self, key: str) -> None:
        """Inject a per-request API key. Used by the BYO-key pattern
        (the request header `X-Provider-Key-<Name>` is forwarded here).

        The key lives only for the lifetime of the current request —
        the backend should call this on every inbound HTTP request.
        """
        self._request_key = (key or "").strip()

    def clear_request_key(self) -> None:
        """Remove any per-request key."""
        self._request_key = ""

    def _key(self) -> str:
        """Return the effective API key.

        Resolution order:
            1. per-request key (set via set_request_key)
            2. SettingsStore (data/settings.json / Postgres in production)
            3. constructor arg + env var (operator-configured fallback)
        """
        if self._request_key:
            return self._request_key
        # Try the persistent store
        try:
            from app.data.settings_store import get_store
            v = get_store().get_provider_key(self.name, request_key="")
            if v:
                return v
        except Exception:
            pass
        return self._constructor_key()

    def _constructor_key(self) -> str:
        """Return the key passed at construction time (or env-var)."""
        return getattr(self, "_static_key", "")

    @abstractmethod
    def is_available(self) -> bool:
        """True if the provider is configured (API key, etc.) and ready to call."""

    @abstractmethod
    def coverage(self) -> dict:
        """Return a dict describing what this provider supports.

        Keys: `markets` (list of "US"/"EU"/"UK"/"CH"/…), `intervals` (list),
        `intraday` (bool), `has_fundamentals` (bool).
        """

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "3mo",
    ) -> Optional[pd.DataFrame]:
        """Return a DataFrame indexed by datetime with columns open/high/low/close/volume,
        or None on failure (raises ProviderError to signal specific failure types)."""

    @abstractmethod
    def fetch_quote(self, symbol: str) -> Optional[float]:
        """Return the current/last price, or None on failure."""

    # ── Shared helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase columns and keep only OHLCV."""
        if df is None or df.empty:
            return df
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        # Some providers return "adj close" — drop it, we want raw close
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill, drop NaN closes, ensure positive prices."""
        if df is None or df.empty:
            return df
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = df.astype(float, errors="ignore")
        if "close" in df.columns:
            df["close"] = df["close"].ffill()
            for col in ("open", "high", "low"):
                if col in df.columns:
                    df[col] = df[col].fillna(df["close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        df = df.dropna(subset=["close"])
        if "close" in df.columns:
            df = df[df["close"] > 0]
        return df

    def __repr__(self) -> str:
        return f"<Provider {self.name} available={self.is_available()}>"
