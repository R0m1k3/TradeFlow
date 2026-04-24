"""
TradeFlow — Base Strategy
Abstract base class for all trading strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class Signal(str, Enum):
    """
    Trading signal emitted by a strategy for a given bar.

    Values:
        BUY: Open or add to a long position.
        SELL: Close or reduce a long position.
        HOLD: Take no action.
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseStrategy(ABC):
    """
    Abstract base class for all TradeFlow trading strategies.

    All concrete strategies must implement:
        - generate_signal(df, current_idx) → tuple[Signal, str]
        - get_params() → dict
        - name property

    The strategy must NOT modify the input DataFrame.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def generate_signal(
        self,
        df: Any,  # pd.DataFrame — avoid import at base level
        current_idx: int,
    ) -> tuple[Signal, str]:
        """
        Evaluate the strategy at a specific bar index and return a trading signal
        along with a human-readable explanation of the decision.

        The strategy must only use data up to and including current_idx
        (no look-ahead bias).

        Args:
            df: OHLCV DataFrame enriched with indicator columns.
            current_idx: Integer position of the current bar in the DataFrame.

        Returns:
            Tuple of (Signal, reason) where reason explains why the signal was emitted.
            Example: (Signal.BUY, "SMA20 (142.50) a croisé au-dessus de SMA50 (140.20)")
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """
        Return the strategy's configurable parameters as a dictionary.

        Returns:
            Dict mapping parameter names to their current values.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.get_params()})"
