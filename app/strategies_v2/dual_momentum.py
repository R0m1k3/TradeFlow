"""
Dual Momentum (Gary Antonacci) — 17.43% CAGR, 22.7% max DD (1974-2013 backtest).

Two conditions must be met for LONG:
  1. Relative Momentum : asset is in the top-K performers of the universe
     over the lookback period (default 12 months).
  2. Absolute Momentum : asset's return > risk-free rate (T-Bills) over the
     same period. If FALSE, switch to cash regardless of relative rank.

Rebalanced monthly.

Reference: Antonacci (2012) "Risk Premia Harvesting Through Dual Momentum"
           SSRN 2042750
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DualSignal(str, Enum):
    LONG = "long"
    CASH = "cash"
    HOLD = "hold"


@dataclass
class DualMomentumPick:
    symbol: str
    momentum_score: float       # return over lookback
    rank: int                    # 1 = best
    signal: DualSignal
    reason: str


class DualMomentumStrategy:
    """
    Screens a universe of symbols. Returns the list of symbols to hold.

    Usage:
        strat = DualMomentumStrategy(top_k=5, lookback_days=252)
        picks = strat.select(prices_dict, risk_free_return)
        for p in picks:
            if p.signal == DualSignal.LONG:
                # route to RiskManager for sizing
    """

    def __init__(
        self,
        top_k: int = 5,
        lookback_days: int = 252,   # 12 months
        min_momentum: float = 0.0,  # absolute momentum threshold above cash
    ) -> None:
        self.top_k = top_k
        self.lookback_days = lookback_days
        self.min_momentum = min_momentum

    def compute_momentum(self, prices: pd.Series) -> float:
        """Total return over lookback window."""
        if len(prices) < self.lookback_days + 1:
            return float("nan")
        start = prices.iloc[-self.lookback_days - 1]
        end = prices.iloc[-1]
        if start <= 0:
            return float("nan")
        return float(end / start - 1)

    def select(
        self,
        prices: dict[str, pd.Series],
        risk_free_return: float = 0.0,
    ) -> list[DualMomentumPick]:
        """
        Args:
            prices: {symbol: close_series} — daily bars, oldest first
            risk_free_return: T-Bill return over the same lookback period

        Returns:
            List of DualMomentumPick, sorted by rank (best first).
            Signal is LONG only if both relative + absolute pass.
        """
        scores = []
        for sym, p in prices.items():
            mom = self.compute_momentum(p)
            if np.isnan(mom):
                continue
            scores.append((sym, mom))

        # Rank descending
        scores.sort(key=lambda x: x[1], reverse=True)

        picks = []
        for rank, (sym, mom) in enumerate(scores, start=1):
            # Absolute momentum: must beat T-Bills (cash)
            beats_cash = mom > (risk_free_return + self.min_momentum)

            # Relative momentum: must be in top-K
            in_top_k = rank <= self.top_k

            if beats_cash and in_top_k:
                signal = DualSignal.LONG
                reason = (
                    f"Dual Momentum OK — rank {rank}/{len(scores)}, "
                    f"return {mom:.1%} > cash {risk_free_return:.1%}"
                )
            elif not beats_cash:
                signal = DualSignal.CASH
                reason = f"Absolute momentum fail ({mom:.1%} ≤ cash {risk_free_return:.1%})"
            else:
                signal = DualSignal.HOLD
                reason = f"Rank {rank} not in top {self.top_k}"

            picks.append(DualMomentumPick(
                symbol=sym,
                momentum_score=mom,
                rank=rank,
                signal=signal,
                reason=reason,
            ))
        return picks

    def is_rebalance_day(self, current_date: pd.Timestamp, last_rebalance: pd.Timestamp | None) -> bool:
        """Monthly rebalancing — first trading day of the month."""
        if last_rebalance is None:
            return True
        return current_date.month != last_rebalance.month or current_date.year != last_rebalance.year
