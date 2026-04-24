"""
TradeFlow — WebUI Metrics Components
Computes and displays Sharpe, drawdown, win rate, and other performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252 * 7,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio from an equity curve.

    Args:
        equity_curve: Pandas Series of portfolio total values indexed by time.
        periods_per_year: Number of bars in a trading year (default: 252*7 for hourly).
        risk_free_rate: Annual risk-free rate as a fraction (default: 0.0).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if insufficient data.
    """
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(round(sharpe, 4))


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Compute the maximum drawdown and its start/end timestamps.

    Args:
        equity_curve: Pandas Series of portfolio total values indexed by time.

    Returns:
        Tuple of (max_drawdown_pct, drawdown_start, drawdown_end).
        Drawdown is expressed as a positive percentage (e.g., 15.2 for -15.2%).
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0, None, None

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100

    max_dd = float(abs(drawdown.min()))
    end_idx = drawdown.idxmin()
    # Find peak before trough
    start_idx = equity_curve[:end_idx].idxmax() if end_idx is not None else None

    return round(max_dd, 4), start_idx, end_idx


def compute_win_rate(trades_df: pd.DataFrame) -> tuple[float, int, int]:
    """
    Compute win rate from a trades DataFrame.

    Args:
        trades_df: DataFrame with at least 'side' and 'pnl' columns.

    Returns:
        Tuple of (win_rate, winning_trades, total_sell_trades).
    """
    if trades_df.empty:
        return 0.0, 0, 0

    sell_trades = trades_df[trades_df["side"] == "SELL"]
    if sell_trades.empty:
        return 0.0, 0, 0

    winning = len(sell_trades[sell_trades["pnl"] > 0])
    total = len(sell_trades)
    win_rate = winning / total if total > 0 else 0.0
    return round(win_rate, 4), winning, total


def compute_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Compute profit factor (gross profit / gross loss) from trades.

    Args:
        trades_df: DataFrame with 'side' and 'pnl' columns.

    Returns:
        Profit factor (> 1 is profitable). Returns 0.0 if no losing trades.
    """
    if trades_df.empty:
        return 0.0

    sell_trades = trades_df[trades_df["side"] == "SELL"]
    gross_profit = sell_trades[sell_trades["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(sell_trades[sell_trades["pnl"] < 0]["pnl"].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return round(gross_profit / gross_loss, 4)


def compute_average_trade_pnl(trades_df: pd.DataFrame) -> float:
    """
    Compute the average P&L per completed (SELL) trade.

    Args:
        trades_df: DataFrame with 'side' and 'pnl' columns.

    Returns:
        Mean P&L per trade in account currency.
    """
    if trades_df.empty:
        return 0.0

    sell_trades = trades_df[trades_df["side"] == "SELL"]
    if sell_trades.empty:
        return 0.0

    return round(sell_trades["pnl"].mean(), 4)
