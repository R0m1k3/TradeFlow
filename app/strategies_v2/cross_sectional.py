"""
Cross-Sectional Momentum — Jegadeesh & Titman (1993), updated by AQR.

Mechanics:
  1. Compute 3-12 month return for every asset (skipping the most recent month
     to avoid short-term reversal — the "12-1" formulation).
  2. Rank all assets.
  3. Long the top decile, optionally short the bottom (omitted here — long-only).
  4. Rebalance monthly.

Empirical: AQR reports ~10.5% / year, 15.3% std-dev on US equities 1963-2016.

Reference:
  - Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
  - AQR, "Fact, Fiction, and Momentum Investing" (2014)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalPick:
    symbol: str
    score: float        # 12-1 month return
    rank: int           # 1 = best
    percentile: float   # 0-1
    in_top_decile: bool


class CrossSectionalMomentumStrategy:
    """
    Ranks all assets by 12-1 momentum. Long the top decile.

    Why 12-1 (not 12-month straight)?
    Short-term reversal: the most recent month's return is negatively correlated
    with next month's. Skipping it improves Sharpe ~20-30% in equities.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_decile: float = 0.10,
        bars_per_month: int = 21,   # ~21 trading days per month
    ) -> None:
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_decile = top_decile
        self.bars_per_month = bars_per_month

    def compute_momentum_12_1(self, prices: pd.Series) -> float:
        """12-1 momentum: (price 21 days ago) / (price 252 days ago) − 1."""
        lookback = self.lookback_months * self.bars_per_month
        skip = self.skip_months * self.bars_per_month

        if len(prices) < lookback + 1:
            return float("nan")

        end_idx = -skip - 1 if skip > 0 else -1
        start_idx = -lookback - 1

        start = prices.iloc[start_idx]
        end = prices.iloc[end_idx]

        if start <= 0:
            return float("nan")
        return float(end / start - 1)

    def rank(self, prices: dict[str, pd.Series]) -> list[CrossSectionalPick]:
        scores = []
        for sym, p in prices.items():
            s = self.compute_momentum_12_1(p)
            if not np.isnan(s):
                scores.append((sym, s))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        n = len(scores)
        top_n = max(1, int(np.ceil(n * self.top_decile)))

        picks = []
        for rank, (sym, score) in enumerate(scores, start=1):
            pct = 1 - (rank - 1) / max(n - 1, 1)
            picks.append(CrossSectionalPick(
                symbol=sym,
                score=score,
                rank=rank,
                percentile=pct,
                in_top_decile=rank <= top_n,
            ))
        return picks

    def longs(self, prices: dict[str, pd.Series]) -> list[str]:
        """Return just the list of symbols to hold long."""
        return [p.symbol for p in self.rank(prices) if p.in_top_decile]
