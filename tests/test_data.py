"""
TradeFlow — Unit Tests: Data Layer
Tests for OHLCV fetcher and technical indicators.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.indicators import add_all_indicators, add_bollinger, add_macd, add_rsi, add_sma


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with 300 bars for indicator testing."""
    import numpy as np

    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = close.clip(1)  # Ensure positive prices

    return pd.DataFrame(
        {
            "open": close * (1 - np.random.uniform(0, 0.005, n)),
            "high": close * (1 + np.random.uniform(0, 0.01, n)),
            "low": close * (1 - np.random.uniform(0, 0.01, n)),
            "close": close,
            "volume": np.random.randint(100_000, 5_000_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """Minimal 10-bar DataFrame (tests edge cases with insufficient data)."""
    import numpy as np

    n = 10
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


# ─── SMA Tests ────────────────────────────────────────────────────────────────

class TestAddSma:
    def test_adds_expected_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_sma(sample_ohlcv, windows=[20, 50])
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns

    def test_does_not_modify_original(self, sample_ohlcv: pd.DataFrame) -> None:
        original_cols = list(sample_ohlcv.columns)
        add_sma(sample_ohlcv, windows=[20])
        assert list(sample_ohlcv.columns) == original_cols

    def test_sma_values_are_numeric(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_sma(sample_ohlcv, windows=[20])
        non_nan = result["sma_20"].dropna()
        assert len(non_nan) > 0
        assert non_nan.dtype == float or pd.api.types.is_float_dtype(non_nan)

    def test_sma_nan_for_insufficient_bars(self, small_ohlcv: pd.DataFrame) -> None:
        """SMA(50) should be entirely NaN for a 10-bar DataFrame."""
        result = add_sma(small_ohlcv, windows=[50])
        assert result["sma_50"].isna().all()

    def test_default_windows(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_sma(sample_ohlcv)
        for w in [20, 50, 200]:
            assert f"sma_{w}" in result.columns


# ─── RSI Tests ────────────────────────────────────────────────────────────────

class TestAddRsi:
    def test_adds_rsi_column(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_rsi(sample_ohlcv, period=14)
        assert "rsi_14" in result.columns

    def test_rsi_range_valid(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_rsi(sample_ohlcv, period=14)
        non_nan = result["rsi_14"].dropna()
        assert (non_nan >= 0).all()
        assert (non_nan <= 100).all()

    def test_does_not_modify_original(self, sample_ohlcv: pd.DataFrame) -> None:
        original_cols = list(sample_ohlcv.columns)
        add_rsi(sample_ohlcv)
        assert list(sample_ohlcv.columns) == original_cols


# ─── MACD Tests ───────────────────────────────────────────────────────────────

class TestAddMacd:
    def test_adds_macd_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_macd(sample_ohlcv)
        assert any(col.startswith("MACD_") for col in result.columns)
        assert any(col.startswith("MACDs_") for col in result.columns)

    def test_macd_not_all_nan(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_macd(sample_ohlcv)
        macd_col = [c for c in result.columns if c.startswith("MACD_") and "MACDs" not in c and "MACDh" not in c]
        assert len(macd_col) > 0
        assert not result[macd_col[0]].dropna().empty


# ─── Bollinger Tests ──────────────────────────────────────────────────────────

class TestAddBollinger:
    def test_adds_bollinger_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_bollinger(sample_ohlcv)
        assert any(col.startswith("BBU_") for col in result.columns)
        assert any(col.startswith("BBL_") for col in result.columns)
        assert any(col.startswith("BBM_") for col in result.columns)

    def test_upper_greater_than_lower(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_bollinger(sample_ohlcv)
        upper = [c for c in result.columns if c.startswith("BBU_")][0]
        lower = [c for c in result.columns if c.startswith("BBL_")][0]
        valid = result[[upper, lower]].dropna()
        assert (valid[upper] >= valid[lower]).all()


# ─── All Indicators ───────────────────────────────────────────────────────────

class TestAddAllIndicators:
    def test_returns_enriched_dataframe(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_all_indicators(sample_ohlcv)
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns
        assert any(col.startswith("MACD_") for col in result.columns)
        assert any(col.startswith("BBU_") for col in result.columns)

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_all_indicators(sample_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_row_count_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_all_indicators(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
