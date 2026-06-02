"""Source router — tries providers in priority order with automatic fallback.

Each provider is wrapped in the resilience layer (circuit breaker + negative
cache + adaptive backoff) so a single dead source doesn't drag down the whole
system.

Priority chain (default)
------------------------
1. MT5          (fastest, paid via broker; usually available in this app)
2. Finnhub      (60 calls/min, US+EU+Asia, free)
3. Twelve Data  (800 calls/day, best EU coverage including .SW/.DE/.PA, free)
4. Alpha Vantage (25/day, US+EU, last-resort free)
5. Yahoo        (most permissive, but unreliable; often the only one left)
6. Stooq        (often blocked, requires API key as of 2026)

The router records which source actually answered, so dashboards can show
"this bar came from Finnhub (MT5 was open)".

Node.js porting notes
---------------------
`SourceRouter` is a class that owns a list of providers and a `resilience_hook`
for each. Port the priority list and the `fetch` method — that's it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.data.providers.base import BaseProvider, ProviderError
from app.data.resilience_hook import ResilienceGuard, for_source, classify_exception

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a router.fetch() call."""
    df: Optional[pd.DataFrame]
    source: str           # which provider actually answered
    tried: list[str]      # providers attempted, in order
    duration_ms: int


class SourceRouter:
    """Routes OHLCV/quote requests through a priority chain of providers."""

    def __init__(
        self,
        providers: list[BaseProvider],
        *,
        priority: list[str] | None = None,
    ) -> None:
        self._providers = {p.name: p for p in providers}
        # Default priority = order in which providers were passed
        self._priority = priority or [p.name for p in providers]

    @classmethod
    def default(cls) -> "SourceRouter":
        """Build the default router with all standard providers (no key required)."""
        # Import here to avoid circular imports
        from app.data.providers.finnhub_provider import FinnhubProvider
        from app.data.providers.alpha_vantage_provider import AlphaVantageProvider
        from app.data.providers.twelve_data_provider import TwelveDataProvider
        from app.data.providers.yahoo_provider import YahooProvider

        providers = [
            FinnhubProvider(),
            TwelveDataProvider(),
            AlphaVantageProvider(),
            YahooProvider(),
        ]
        return cls(providers, priority=["finnhub", "twelve_data", "alpha_vantage", "yahoo"])

    def available_sources(self) -> list[str]:
        return [name for name in self._priority if self._providers.get(name, _Stub()).is_available()]

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "3mo",
        *,
        prefer: list[str] | None = None,
        keys: dict[str, str] | None = None,
    ) -> FetchResult:
        """Try each provider in priority order. Return the first success.

        `keys` is an optional dict {provider_name: api_key} that overrides
        the per-provider env-var key for this single call (BYO-key pattern).
        Pass `None` to fall back to env-var keys.
        """
        started = time.time()
        order = prefer or self._priority
        tried: list[str] = []
        keys = keys or {}

        for name in order:
            provider = self._providers.get(name)
            if provider is None:
                continue

            # Inject per-request key (if any), then check availability
            if name in keys and keys[name]:
                provider.set_request_key(keys[name])
            else:
                provider.clear_request_key()

            if not provider.is_available():
                continue

            guard = for_source(name)
            key = f"{symbol}:{interval}:{period}"
            decision = guard.before_call(key)
            if not decision.proceed:
                logger.debug("Router: skipping %s (%s)", name, decision.reason)
                tried.append(name)
                continue

            tried.append(name)
            try:
                df = provider.fetch_ohlcv(symbol, interval, period)
                guard.after_success(key)
                duration_ms = int((time.time() - started) * 1000)
                if df is not None and not df.empty:
                    logger.info(
                        "Router: %s returned %d bars for %s [%s] in %dms (tried %s)",
                        name, len(df), symbol, interval, duration_ms, tried,
                    )
                    return FetchResult(df=df, source=name, tried=tried, duration_ms=duration_ms)
            except ProviderError as exc:
                kind = exc.kind
                guard.after_failure(key, kind=kind)
                logger.warning(
                    "Router: %s failed for %s [%s] — %s (%s)",
                    name, symbol, interval, exc, kind,
                )
                continue
            except Exception as exc:
                kind = classify_exception(exc)
                guard.after_failure(key, kind=kind)
                logger.warning(
                    "Router: %s raised for %s [%s] — %s (classified as %s)",
                    name, symbol, interval, exc, kind,
                )
                continue

        duration_ms = int((time.time() - started) * 1000)
        logger.warning(
            "Router: NO source returned data for %s [%s] (tried %s in %dms)",
            symbol, interval, tried, duration_ms,
        )
        return FetchResult(df=None, source="", tried=tried, duration_ms=duration_ms)

    def fetch_quote(
        self,
        symbol: str,
        *,
        prefer: list[str] | None = None,
        keys: dict[str, str] | None = None,
    ) -> FetchResult:
        """Like fetch_ohlcv but returns the current price. Accepts `keys`."""
        started = time.time()
        order = prefer or self._priority
        tried: list[str] = []
        keys = keys or {}

        for name in order:
            provider = self._providers.get(name)
            if provider is None:
                continue
            if name in keys and keys[name]:
                provider.set_request_key(keys[name])
            else:
                provider.clear_request_key()
            if not provider.is_available():
                continue

            guard = for_source(name)
            key = f"{symbol}:quote"
            decision = guard.before_call(key)
            if not decision.proceed:
                tried.append(name)
                continue

            tried.append(name)
            try:
                price = provider.fetch_quote(symbol)
                guard.after_success(key)
                if price is not None:
                    duration_ms = int((time.time() - started) * 1000)
                    logger.info("Router: %s returned quote %.2f for %s", name, price, symbol)
                    return FetchResult(
                        df=pd.DataFrame({"close": [price]}, index=[pd.Timestamp.now("UTC")]),
                        source=name, tried=tried, duration_ms=duration_ms,
                    )
            except ProviderError as exc:
                guard.after_failure(key, kind=exc.kind)
                continue
            except Exception as exc:
                kind = classify_exception(exc)
                guard.after_failure(key, kind=kind)
                continue

        duration_ms = int((time.time() - started) * 1000)
        return FetchResult(df=None, source="", tried=tried, duration_ms=duration_ms)

    def set_request_keys(self, keys: dict[str, str]) -> None:
        """Inject per-request keys into all known providers at once."""
        for name, provider in self._providers.items():
            if name in keys and keys[name]:
                provider.set_request_key(keys[name])
            else:
                provider.clear_request_key()

    def stats(self) -> dict:
        return {
            "providers": [self._providers[n].__dict__ if n in self._providers else None
                          for n in self._priority],
        }


# Helper: a stub provider for `is_available` checks on unknown names
class _Stub(BaseProvider):
    name = "_stub"
    def is_available(self) -> bool: return False
    def coverage(self) -> dict: return {}
    def fetch_ohlcv(self, *a, **kw): return None
    def fetch_quote(self, *a, **kw): return None
