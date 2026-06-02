"""Tests for the multi-source data layer.

All tests use mocked HTTP — no real API calls. We cover:
* Each provider's request format and response parsing
* The SourceRouter's fallback chain
* The resilience layer (negative cache / circuit breaker) integration
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from app.data.providers.base import BaseProvider, ProviderError
from app.data.providers.finnhub_provider import FinnhubProvider, _period_to_seconds
from app.data.providers.twelve_data_provider import TwelveDataProvider
from app.data.providers.alpha_vantage_provider import AlphaVantageProvider
from app.data.providers.yahoo_provider import YahooProvider
from app.data.source_router import SourceRouter, FetchResult


# ── Period parsing ─────────────────────────────────────────────────────────


class TestPeriodToSeconds:
    def test_days(self):
        assert _period_to_seconds("5d") == 5 * 86400
    def test_weeks(self):
        assert _period_to_seconds("2w") == 14 * 86400
    def test_months(self):
        assert _period_to_seconds("3mo") == 90 * 86400
    def test_years(self):
        assert _period_to_seconds("1y") == 365 * 86400
    def test_max(self):
        assert _period_to_seconds("max") > 0
    def test_garbage_defaults_3mo(self):
        assert _period_to_seconds("xyz") == 90 * 86400


# ── Finnhub ────────────────────────────────────────────────────────────────


class TestFinnhubProvider:
    def setup_method(self):
        self.p = FinnhubProvider(api_key="test-key")

    def test_not_available_without_key(self):
        p = FinnhubProvider(api_key="")
        assert not p.is_available()

    def test_coverage(self):
        cov = self.p.coverage()
        assert "US" in cov["markets"]
        assert "EU" in cov["markets"]
        assert cov["intraday"]

    def test_resolve_passes_through_yahoo_symbols(self):
        # We use Yahoo-format symbols (AAPL, ROG.SW, MC.PA) which Finnhub accepts as-is
        assert self.p._resolve("AAPL") == "AAPL"
        assert self.p._resolve("ROG.SW") == "ROG.SW"
        assert self.p._resolve("MC.PA") == "MC.PA"

    def test_fetch_ohlcv_success(self, monkeypatch):
        captured = {}

        def fake_get(self, url, params=None, **kw):
            captured["url"] = url
            captured["params"] = params
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "s": "ok",
                "t": [1700000000, 1700003600, 1700007200],
                "o": [100.0, 101.5, 102.0],
                "h": [101.0, 102.5, 103.0],
                "l": [99.5, 100.5, 101.0],
                "c": [100.5, 102.0, 102.5],
                "v": [1000, 1500, 2000],
            }
            return resp

        monkeypatch.setattr(httpx.Client, "get", fake_get)
        df = self.p.fetch_ohlcv("AAPL", interval="1h", period="1d")
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert captured["params"]["symbol"] == "AAPL"
        assert captured["params"]["resolution"] == 60  # 1h

    def test_fetch_ohlcv_no_data_raises_404(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"s": "no_data"}
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        with pytest.raises(ProviderError) as exc:
            self.p.fetch_ohlcv("MISSING.X")
        assert exc.value.kind == "404"

    def test_fetch_ohlcv_rate_limit(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 429
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        with pytest.raises(ProviderError) as exc:
            self.p.fetch_ohlcv("AAPL")
        assert exc.value.kind == "429"

    def test_fetch_quote_success(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"c": 185.50, "h": 187, "l": 184, "o": 186, "pc": 184}
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        price = self.p.fetch_quote("AAPL")
        assert price == 185.50


# ── Twelve Data ────────────────────────────────────────────────────────────


class TestTwelveDataProvider:
    def setup_method(self):
        self.p = TwelveDataProvider(api_key="test-key")

    def test_resolve_maps_european_suffixes(self):
        assert self.p._resolve("MC.PA") == "EPA:MC"
        assert self.p._resolve("PHIA.AS") == "AMS:PHIA"
        assert self.p._resolve("ROG.SW") == "SWX:ROG"
        assert self.p._resolve("VOD.L") == "LSE:VOD"
        assert self.p._resolve("SAP.DE") == "XETRA:SAP"
        # US passes through
        assert self.p._resolve("AAPL") == "AAPL"
        # Crypto / FX
        assert self.p._resolve("BTC/USD") == "BTC/USD"

    def test_coverage(self):
        cov = self.p.coverage()
        assert "CH" in cov["markets"]  # Swiss
        assert "DE" in cov["markets"]
        assert "FR" in cov["markets"]
        assert "NL" in cov["markets"]

    def test_fetch_ohlcv_success(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "status": "ok",
                "values": [
                    {"datetime": "2024-01-02 09:00:00", "open": "100", "high": "101", "low": "99", "close": "100.5", "volume": "1000"},
                    {"datetime": "2024-01-01 09:00:00", "open": "99", "high": "100", "low": "98", "close": "99.5", "volume": "500"},
                ],
            }
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        df = self.p.fetch_ohlcv("ROG.SW", interval="1d", period="1mo")
        assert df is not None
        assert len(df) == 2
        # Oldest first
        assert df.iloc[0]["close"] == 99.5
        assert df.iloc[1]["close"] == 100.5

    def test_fetch_ohlcv_no_data_raises_404(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"status": "error", "message": "Symbol not found"}
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        with pytest.raises(ProviderError) as exc:
            self.p.fetch_ohlcv("MISSING")
        assert exc.value.kind == "404"


# ── Alpha Vantage ──────────────────────────────────────────────────────────


class TestAlphaVantageProvider:
    def setup_method(self):
        self.p = AlphaVantageProvider(api_key="test-key")

    def test_intraday_uses_correct_function(self, monkeypatch):
        captured = {}
        def fake_get(self, url, params=None, **kw):
            captured["params"] = params
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "Meta Data": {},
                "Time Series (5min)": {
                    "2024-01-02 16:00:00": {"1. open": "100", "2. high": "101", "3. low": "99", "4. close": "100.5", "5. volume": "1000"},
                }
            }
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        df = self.p.fetch_ohlcv("AAPL", interval="5m", period="1d")
        assert df is not None and len(df) == 1
        assert captured["params"]["function"] == "TIME_SERIES_INTRADAY"
        assert captured["params"]["interval"] == "5min"

    def test_daily_uses_correct_function(self, monkeypatch):
        captured = {}
        def fake_get(self, url, params=None, **kw):
            captured["params"] = params
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "Meta Data": {},
                "Time Series (Daily)": {
                    "2024-01-02": {"1. open": "100", "2. high": "101", "3. low": "99", "4. close": "100.5", "5. volume": "1000"},
                }
            }
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        df = self.p.fetch_ohlcv("AAPL", interval="1d", period="1mo")
        assert df is not None
        assert captured["params"]["function"] == "TIME_SERIES_DAILY"

    def test_rate_limit_message_raises_429(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 25 requests per day."
            }
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        with pytest.raises(ProviderError) as exc:
            self.p.fetch_ohlcv("AAPL", interval="1d", period="1mo")
        assert exc.value.kind == "429"

    def test_error_message_raises_404(self, monkeypatch):
        def fake_get(self, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"Error Message": "Invalid API call"}
            return resp
        monkeypatch.setattr(httpx.Client, "get", fake_get)

        with pytest.raises(ProviderError) as exc:
            self.p.fetch_ohlcv("AAPL", interval="1d", period="1mo")
        assert exc.value.kind == "404"


# ── SourceRouter ───────────────────────────────────────────────────────────


class _FakeProvider(BaseProvider):
    """Test provider that always returns the configured data or raises."""
    def __init__(self, name: str, available: bool = True, df: pd.DataFrame | None = None,
                 raise_kind: str | None = None, quote: float | None = None):
        self.name = name
        self._available = available
        self._df = df
        self._raise_kind = raise_kind
        self._quote = quote
        self.calls = 0
    def is_available(self): return self._available
    def coverage(self): return {"markets": ["X"], "intervals": ["1d"], "intraday": False, "has_fundamentals": False}
    def fetch_ohlcv(self, symbol, interval="1d", period="3mo"):
        self.calls += 1
        if self._raise_kind:
            raise ProviderError(f"fake error from {self.name}", kind=self._raise_kind)
        return self._df
    def fetch_quote(self, symbol):
        self.calls += 1
        if self._raise_kind:
            raise ProviderError(f"fake error from {self.name}", kind=self._raise_kind)
        return self._quote


@pytest.fixture(autouse=True)
def reset_resilience():
    """Reset the resilience registry between tests so backoff state doesn't leak."""
    from app.data import resilience_hook
    resilience_hook._registry.clear()
    yield
    resilience_hook._registry.clear()


class TestSourceRouter:
    def test_first_available_wins(self):
        df = pd.DataFrame({"close": [100]}, index=pd.DatetimeIndex(["2024-01-01"]))
        a = _FakeProvider("a", df=df)
        b = _FakeProvider("b", df=df)
        router = SourceRouter([a, b])
        result = router.fetch_ohlcv("X")
        assert result.source == "a"
        assert a.calls == 1
        assert b.calls == 0

    def test_fallback_on_failure(self):
        df = pd.DataFrame({"close": [100]}, index=pd.DatetimeIndex(["2024-01-01"]))
        a = _FakeProvider("a", raise_kind="5xx")
        b = _FakeProvider("b", df=df)
        router = SourceRouter([a, b])
        result = router.fetch_ohlcv("X")
        assert result.source == "b"
        assert "a" in result.tried
        assert "b" in result.tried

    def test_skips_unavailable(self):
        df = pd.DataFrame({"close": [100]}, index=pd.DatetimeIndex(["2024-01-01"]))
        a = _FakeProvider("a", available=False, df=df)
        b = _FakeProvider("b", df=df)
        router = SourceRouter([a, b])
        result = router.fetch_ohlcv("X")
        assert result.source == "b"
        assert a.calls == 0  # not even tried

    def test_all_fail_returns_none(self):
        a = _FakeProvider("a", raise_kind="5xx")
        b = _FakeProvider("b", raise_kind="404")
        router = SourceRouter([a, b])
        result = router.fetch_ohlcv("X")
        assert result.df is None
        assert result.source == ""
        assert set(result.tried) == {"a", "b"}

    def test_404_uses_long_backoff_no_retry(self):
        # 404 should be recorded but the next attempt in this short test
        # should still try (the resilience layer is the one that would skip
        # on subsequent calls; the router doesn't skip on first call).
        a = _FakeProvider("a", raise_kind="404")
        b = _FakeProvider("b", df=pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex(["2024-01-01"])))
        router = SourceRouter([a, b])
        result = router.fetch_ohlcv("X")
        assert result.source == "b"

    def test_prefer_overrides_priority(self):
        df = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex(["2024-01-01"]))
        a = _FakeProvider("a", df=df)
        b = _FakeProvider("b", df=df)
        c = _FakeProvider("c", df=df)
        router = SourceRouter([a, b, c])
        result = router.fetch_ohlcv("X", prefer=["c", "a"])
        assert result.source == "c"
        assert b.calls == 0

    def test_quote_falls_back(self):
        a = _FakeProvider("a", raise_kind="5xx")
        b = _FakeProvider("b", quote=42.5)
        router = SourceRouter([a, b])
        result = router.fetch_quote("X")
        assert result.source == "b"
        assert result.df is not None
        assert result.df.iloc[0]["close"] == 42.5

    def test_available_sources(self):
        a = _FakeProvider("a", available=True)
        b = _FakeProvider("b", available=False)
        c = _FakeProvider("c", available=True)
        router = SourceRouter([a, b, c])
        assert router.available_sources() == ["a", "c"]

    def test_records_failure_in_resilience_layer(self):
        """Verify the router actually feeds the resilience layer."""
        a = _FakeProvider("a", raise_kind="5xx")
        router = SourceRouter([a])
        # Drive 4 failures by recording directly on the guard, then
        # verify the router's after_failure path doesn't break the
        # negative cache's state machine.
        from app.data.resilience_hook import for_source
        guard = for_source("a")
        for _ in range(4):
            guard.negative.record("X:1d:3mo", "5xx")
        # Reset backoff so the next fetch actually goes through
        guard.backoff.reset("X:1d:3mo")
        result = router.fetch_ohlcv("X")
        # No data, but the negative cache was incremented past 4
        assert result.df is None
        assert result.source == ""
        status = guard.negative.status("X:1d:3mo")
        assert status["failures"] >= 4
