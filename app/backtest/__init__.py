"""Backtesting harness — walk-forward + CPCV + Deflated Sharpe."""
from app.backtest.walkforward import WalkForward, WalkForwardResult
from app.backtest.cpcv import CombinatorialPurgedCV
from app.backtest.metrics import (
    sharpe_ratio,
    deflated_sharpe_ratio,
    max_drawdown,
    profit_factor,
    win_rate,
    calmar_ratio,
    summary_stats,
)

__all__ = [
    "WalkForward",
    "WalkForwardResult",
    "CombinatorialPurgedCV",
    "sharpe_ratio",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "win_rate",
    "calmar_ratio",
    "summary_stats",
]
