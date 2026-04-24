"""
TradeFlow — SMA Crossover Strategy
Generates BUY/SELL signals based on fast/slow SMA crossover.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class SmaCrossoverStrategy(BaseStrategy):
    """
    SMA Crossover Strategy.

    Logic:
        - BUY  when fast SMA crosses ABOVE slow SMA (golden cross).
        - SELL when fast SMA crosses BELOW slow SMA (death cross).
        - HOLD otherwise.

    Requires DataFrame to contain columns: sma_{fast_period}, sma_{slow_period}.
    Use app.data.indicators.add_sma() to enrich the DataFrame before simulation.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        """
        Initialize strategy with configurable SMA periods.

        Args:
            fast_period: Lookback window for the fast SMA (default: 20).
            slow_period: Lookback window for the slow SMA (default: 50).

        Raises:
            ValueError: If fast_period >= slow_period.
        """
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )
        self._fast_period = fast_period
        self._slow_period = slow_period

    @property
    def name(self) -> str:
        return f"SMA Crossover ({self._fast_period}/{self._slow_period})"

    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> tuple[Signal, str]:
        """
        Evaluate SMA crossover at the given bar index.

        Args:
            df: OHLCV DataFrame with sma_{fast} and sma_{slow} columns.
            current_idx: Current bar index (0-based).

        Returns:
            Tuple of (Signal, reason_str).
        """
        fast_col = f"sma_{self._fast_period}"
        slow_col = f"sma_{self._slow_period}"

        if current_idx < 1:
            return Signal.HOLD, ""

        if fast_col not in df.columns or slow_col not in df.columns:
            logger.warning(
                "Missing SMA columns '%s' or '%s'. Run add_sma() first.", fast_col, slow_col
            )
            return Signal.HOLD, f"Colonnes SMA manquantes ({fast_col}, {slow_col})"

        fast_now = df[fast_col].iloc[current_idx]
        fast_prev = df[fast_col].iloc[current_idx - 1]
        slow_now = df[slow_col].iloc[current_idx]
        slow_prev = df[slow_col].iloc[current_idx - 1]

        if any(pd.isna(v) for v in [fast_now, fast_prev, slow_now, slow_prev]):
            return Signal.HOLD, ""

        # Golden cross: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            reason = (
                f"Croisement haussier (Golden Cross) : "
                f"SMA{self._fast_period} ({fast_now:.2f}) a croisé au-dessus de "
                f"SMA{self._slow_period} ({slow_now:.2f})"
            )
            logger.debug("BUY signal at idx=%d (golden cross)", current_idx)
            return Signal.BUY, reason

        # Death cross: fast crosses below slow
        if fast_prev >= slow_prev and fast_now < slow_now:
            reason = (
                f"Croisement baissier (Death Cross) : "
                f"SMA{self._fast_period} ({fast_now:.2f}) a croisé en-dessous de "
                f"SMA{self._slow_period} ({slow_now:.2f})"
            )
            logger.debug("SELL signal at idx=%d (death cross)", current_idx)
            return Signal.SELL, reason

        return Signal.HOLD, ""

    def get_params(self) -> dict[str, Any]:
        return {
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
        }
