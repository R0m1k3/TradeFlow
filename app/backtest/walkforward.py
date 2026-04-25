"""
Walk-Forward Validation — the gold standard for time-series strategies.

Train on a rolling window, test on the next out-of-sample period, then roll
forward. Always respects the arrow of time. No look-ahead.

Two modes:
  - Anchored (expanding window): train always starts at t=0, grows over time
  - Rolling: train window has fixed length, slides forward
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold: int


@dataclass
class WalkForwardResult:
    splits: list[WalkForwardSplit] = field(default_factory=list)
    fold_returns: list[pd.Series] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)

    def combined_returns(self) -> pd.Series:
        if not self.fold_returns:
            return pd.Series(dtype=float)
        return pd.concat(self.fold_returns).sort_index()

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s, m in zip(self.splits, self.fold_metrics):
            rows.append({
                "fold": s.fold,
                "train_start": s.train_start,
                "train_end": s.train_end,
                "test_start": s.test_start,
                "test_end": s.test_end,
                **m,
            })
        return pd.DataFrame(rows)


class WalkForward:
    """Generate train/test splits and run a strategy callback over each."""

    def __init__(
        self,
        train_periods: int = 252 * 2,    # 2 years
        test_periods: int = 63,           # 3 months
        embargo: int = 5,                 # purge gap (avoid leakage at boundary)
        anchored: bool = False,
    ) -> None:
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.embargo = embargo
        self.anchored = anchored

    def split(self, index: pd.DatetimeIndex) -> Iterator[WalkForwardSplit]:
        n = len(index)
        if n < self.train_periods + self.test_periods + self.embargo:
            return

        fold = 0
        train_start_idx = 0
        train_end_idx = self.train_periods
        while train_end_idx + self.embargo + self.test_periods <= n:
            test_start_idx = train_end_idx + self.embargo
            test_end_idx = min(test_start_idx + self.test_periods, n)

            yield WalkForwardSplit(
                train_start=index[train_start_idx],
                train_end=index[train_end_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
                fold=fold,
            )

            fold += 1
            if self.anchored:
                # Expanding window: keep train_start at 0
                train_end_idx = test_end_idx
            else:
                # Rolling window: slide both
                train_start_idx = test_start_idx
                train_end_idx = train_start_idx + self.train_periods

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
        metrics_fn: Callable[[pd.Series], dict] | None = None,
    ) -> WalkForwardResult:
        """
        Args:
            data: full-history dataframe indexed by datetime
            strategy_fn: (train_df, test_df) → returns series indexed by test dates
            metrics_fn: returns → dict of metrics (default: summary_stats)

        Returns:
            WalkForwardResult with per-fold returns + metrics
        """
        from app.backtest.metrics import summary_stats

        if metrics_fn is None:
            metrics_fn = summary_stats

        result = WalkForwardResult()
        for split in self.split(data.index):
            train = data.loc[split.train_start:split.train_end]
            test = data.loc[split.test_start:split.test_end]
            try:
                returns = strategy_fn(train, test)
                metrics = metrics_fn(returns)
                result.splits.append(split)
                result.fold_returns.append(returns)
                result.fold_metrics.append(metrics)
                logger.info(
                    "Fold %d | %s → %s | sharpe=%.2f maxDD=%.2f",
                    split.fold,
                    split.test_start.date(),
                    split.test_end.date(),
                    metrics.get("sharpe", 0),
                    metrics.get("max_drawdown", 0),
                )
            except Exception as exc:
                logger.error("Fold %d failed: %s", split.fold, exc, exc_info=True)
        return result
