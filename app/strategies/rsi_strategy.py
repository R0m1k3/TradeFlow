"""
TradeFlow — RSI Strategy
Generates BUY/SELL signals based on RSI overbought/oversold levels.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class RsiStrategy(BaseStrategy):
    """
    RSI Oversold/Overbought Strategy.

    Logic:
        - BUY  when RSI crosses UP through the oversold threshold (e.g., RSI < 30).
        - SELL when RSI crosses DOWN through the overbought threshold (e.g., RSI > 70).
        - HOLD otherwise.

    Requires DataFrame to contain column: rsi_{period}.
    Use app.data.indicators.add_rsi() to enrich the DataFrame before simulation.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        """
        Initialize strategy with configurable RSI parameters.

        Args:
            period: RSI calculation period (default: 14).
            oversold: RSI threshold below which market is oversold (default: 30).
            overbought: RSI threshold above which market is overbought (default: 70).

        Raises:
            ValueError: If oversold >= overbought.
        """
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI ({self._period}) [{self._oversold}/{self._overbought}]"

    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> tuple[Signal, str]:
        """
        Evaluate RSI signal at the given bar index.

        Uses crossover detection to avoid repeated signals in the same zone.

        Args:
            df: OHLCV DataFrame with rsi_{period} column.
            current_idx: Current bar index (0-based).

        Returns:
            Tuple of (Signal, reason_str).
        """
        rsi_col = f"rsi_{self._period}"

        if current_idx < 1:
            return Signal.HOLD, ""

        if rsi_col not in df.columns:
            logger.warning("Missing RSI column '%s'. Run add_rsi() first.", rsi_col)
            return Signal.HOLD, f"Colonne RSI manquante ({rsi_col})"

        rsi_now = df[rsi_col].iloc[current_idx]
        rsi_prev = df[rsi_col].iloc[current_idx - 1]

        if pd.isna(rsi_now) or pd.isna(rsi_prev):
            return Signal.HOLD, ""

        # RSI crosses up from below oversold threshold → BUY
        if rsi_prev <= self._oversold and rsi_now > self._oversold:
            reason = (
                f"RSI survendu : RSI={rsi_now:.1f} a croisé le seuil {self._oversold} à la hausse "
                f"(était {rsi_prev:.1f}) — rebond probable"
            )
            logger.debug("BUY signal at idx=%d (RSI crossover up, RSI=%.2f)", current_idx, rsi_now)
            return Signal.BUY, reason

        # RSI crosses down from above overbought threshold → SELL
        if rsi_prev >= self._overbought and rsi_now < self._overbought:
            reason = (
                f"RSI suracheté : RSI={rsi_now:.1f} a croisé le seuil {self._overbought} à la baisse "
                f"(était {rsi_prev:.1f}) — correction probable"
            )
            logger.debug("SELL signal at idx=%d (RSI crossover down, RSI=%.2f)", current_idx, rsi_now)
            return Signal.SELL, reason

        return Signal.HOLD, ""

    def get_params(self) -> dict[str, Any]:
        return {
            "period": self._period,
            "oversold": self._oversold,
            "overbought": self._overbought,
        }
