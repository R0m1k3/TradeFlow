"""
TradeFlow — Unit Tests: Strategies
Tests for SMA Crossover, RSI, and MACD strategy signal generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.indicators import add_all_indicators, add_macd, add_rsi, add_sma
from app.strategies.base import Signal
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.sma_crossover import SmaCrossoverStrategy


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_crossover_df(n: int = 100) -> pd.DataFrame:
    """
    Create a DataFrame designed to trigger a golden cross at bar 60.
    Fast SMA > Slow SMA after bar 60.
    """
    close = np.ones(n) * 100.0
    # Create an upward spike to trigger crossover
    close[55:] += np.linspace(0, 30, n - 55)

    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    return add_sma(df, windows=[20, 50])


def make_rsi_df(n: int = 100) -> pd.DataFrame:
    """
    Create a DataFrame with RSI that dips below 30 around bar 30.
    """
    close = np.ones(n) * 100.0
    # Rapid decline → low RSI
    close[20:35] = np.linspace(100, 60, 15)
    close[35:] = np.linspace(60, 85, n - 35)

    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    return add_rsi(df, period=14)


def make_macd_df(n: int = 200) -> pd.DataFrame:
    """Create a DataFrame with enough bars for valid MACD signals."""
    np.random.seed(99)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = close.clip(1)

    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    return add_macd(df, fast=12, slow=26, signal=9)


# ─── SmaCrossoverStrategy Tests ───────────────────────────────────────────────

class TestSmaCrossoverStrategy:
    def test_name(self) -> None:
        strat = SmaCrossoverStrategy(fast_period=20, slow_period=50)
        assert "SMA" in strat.name
        assert "20" in strat.name
        assert "50" in strat.name

    def test_get_params(self) -> None:
        strat = SmaCrossoverStrategy(fast_period=10, slow_period=30)
        params = strat.get_params()
        assert params["fast_period"] == 10
        assert params["slow_period"] == 30

    def test_invalid_periods_raises(self) -> None:
        with pytest.raises(ValueError):
            SmaCrossoverStrategy(fast_period=50, slow_period=20)

    def test_hold_on_first_bar(self) -> None:
        df = make_crossover_df()
        strat = SmaCrossoverStrategy()
        assert strat.generate_signal(df, 0) == Signal.HOLD

    def test_hold_when_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [100.0] * 50})
        strat = SmaCrossoverStrategy()
        assert strat.generate_signal(df, 25) == Signal.HOLD

    def test_signals_are_valid_enum(self) -> None:
        df = make_crossover_df(100)
        strat = SmaCrossoverStrategy(fast_period=20, slow_period=50)
        for i in range(1, len(df)):
            sig = strat.generate_signal(df, i)
            assert sig in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_generates_some_non_hold_signal(self) -> None:
        df = make_crossover_df(100)
        strat = SmaCrossoverStrategy(fast_period=20, slow_period=50)
        signals = [strat.generate_signal(df, i) for i in range(1, len(df))]
        non_hold = [s for s in signals if s != Signal.HOLD]
        # Should have at least one BUY or SELL in this engineered data
        assert len(non_hold) >= 1


# ─── RsiStrategy Tests ────────────────────────────────────────────────────────

class TestRsiStrategy:
    def test_name(self) -> None:
        strat = RsiStrategy(period=14, oversold=30, overbought=70)
        assert "RSI" in strat.name

    def test_get_params(self) -> None:
        strat = RsiStrategy(period=7, oversold=25, overbought=75)
        params = strat.get_params()
        assert params["period"] == 7
        assert params["oversold"] == 25
        assert params["overbought"] == 75

    def test_invalid_thresholds_raises(self) -> None:
        with pytest.raises(ValueError):
            RsiStrategy(oversold=70, overbought=30)

    def test_hold_on_first_bar(self) -> None:
        df = make_rsi_df()
        strat = RsiStrategy()
        assert strat.generate_signal(df, 0) == Signal.HOLD

    def test_hold_when_missing_rsi_column(self) -> None:
        df = pd.DataFrame({"close": [100.0] * 30})
        strat = RsiStrategy()
        assert strat.generate_signal(df, 15) == Signal.HOLD

    def test_signals_are_valid_enum(self) -> None:
        df = make_rsi_df(100)
        strat = RsiStrategy(period=14)
        for i in range(1, len(df)):
            sig = strat.generate_signal(df, i)
            assert sig in (Signal.BUY, Signal.SELL, Signal.HOLD)


# ─── MacdStrategy Tests ───────────────────────────────────────────────────────

class TestMacdStrategy:
    def test_name(self) -> None:
        strat = MacdStrategy()
        assert "MACD" in strat.name

    def test_get_params(self) -> None:
        strat = MacdStrategy(fast=12, slow=26, signal=9)
        params = strat.get_params()
        assert params["fast"] == 12
        assert params["slow"] == 26
        assert params["signal"] == 9

    def test_hold_on_first_bar(self) -> None:
        df = make_macd_df()
        strat = MacdStrategy()
        assert strat.generate_signal(df, 0) == Signal.HOLD

    def test_hold_when_missing_macd_columns(self) -> None:
        df = pd.DataFrame({"close": [100.0] * 50})
        strat = MacdStrategy()
        assert strat.generate_signal(df, 25) == Signal.HOLD

    def test_signals_are_valid_enum(self) -> None:
        df = make_macd_df(200)
        strat = MacdStrategy()
        for i in range(1, len(df)):
            sig = strat.generate_signal(df, i)
            assert sig in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_generates_some_signals(self) -> None:
        df = make_macd_df(200)
        strat = MacdStrategy()
        signals = [strat.generate_signal(df, i) for i in range(1, len(df))]
        non_hold = [s for s in signals if s != Signal.HOLD]
        assert len(non_hold) >= 1
