"""
Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado, 2018.

Standard k-fold CV leaks information in time series. Purged k-fold removes
overlapping training samples around each test fold. CPCV goes further by
generating *combinations* of test folds, producing many more backtest paths
from the same data, which gives a much more robust estimate of out-of-sample
performance and reduces backtest overfitting.

Reference:
  Lopez de Prado, "Advances in Financial Machine Learning" (2018), ch.12
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_groups: tuple[int, ...]
    path_id: int


class CombinatorialPurgedCV:
    """
    Splits the time index into N groups, then for each combination of K
    "test" groups, builds a (purged) train set from the remaining (N-K) groups.

    Number of paths generated = C(N, K).

    Common settings (Lopez de Prado): N=10 groups, K=2 test → 45 paths.
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, index: pd.DatetimeIndex) -> Iterator[CPCVSplit]:
        n = len(index)
        if n < self.n_groups * 2:
            return

        group_size = n // self.n_groups
        embargo = int(n * self.embargo_pct)

        # Build the boundary indices of each group
        group_bounds = [(i * group_size, (i + 1) * group_size) for i in range(self.n_groups)]
        # Last group absorbs the remainder
        last_start, _ = group_bounds[-1]
        group_bounds[-1] = (last_start, n)

        path_id = 0
        for test_combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_indices_list = []
            for g in test_combo:
                start, end = group_bounds[g]
                test_indices_list.extend(range(start, end))
            test_indices = np.array(sorted(test_indices_list))

            # Train = all indices NOT in test, with embargo around each test boundary
            forbidden = set(test_indices.tolist())
            for g in test_combo:
                start, end = group_bounds[g]
                # Embargo before and after each test group
                for j in range(max(0, start - embargo), start):
                    forbidden.add(j)
                for j in range(end, min(n, end + embargo)):
                    forbidden.add(j)

            train_indices = np.array([i for i in range(n) if i not in forbidden])

            yield CPCVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                test_groups=test_combo,
                path_id=path_id,
            )
            path_id += 1

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    ) -> dict:
        """
        Run a strategy over all CPCV paths.

        Returns:
            dict with:
              - 'paths_returns': list of return series, one per path
              - 'sharpe_distribution': array of Sharpe ratios across paths
              - 'pbo': probability of backtest overfitting estimate
        """
        from app.backtest.metrics import sharpe_ratio

        path_returns = []
        sharpes = []

        for split in self.split(data.index):
            train = data.iloc[split.train_indices]
            test = data.iloc[split.test_indices]
            try:
                rets = strategy_fn(train, test)
                path_returns.append(rets)
                sharpes.append(sharpe_ratio(rets))
            except Exception as exc:
                logger.warning("CPCV path %d failed: %s", split.path_id, exc)

        sharpes_arr = np.array(sharpes)
        # Probability of backtest overfitting: rough proxy as fraction with negative Sharpe
        pbo = float((sharpes_arr <= 0).mean()) if len(sharpes_arr) else 1.0

        return {
            "paths_returns": path_returns,
            "sharpe_distribution": sharpes_arr,
            "median_sharpe": float(np.median(sharpes_arr)) if len(sharpes_arr) else 0.0,
            "mean_sharpe": float(np.mean(sharpes_arr)) if len(sharpes_arr) else 0.0,
            "std_sharpe": float(np.std(sharpes_arr)) if len(sharpes_arr) else 0.0,
            "n_paths": len(path_returns),
            "pbo": pbo,
        }
