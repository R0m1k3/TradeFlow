"""
Performance metrics — including Deflated Sharpe (Lopez de Prado, 2014).

Why Deflated Sharpe matters:
  Running 100 backtests, picking the best-Sharpe one, and reporting THAT Sharpe
  is a textbook example of multiple-testing bias. The Deflated Sharpe Ratio (DSR)
  corrects for the number of trials and the higher moments (skew, kurtosis) of
  the strategy returns, giving the probability that the observed Sharpe is real.

Reference:
  Lopez de Prado & Bailey (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality"
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (assumes risk-free = 0)."""
    r = _safe_returns(returns)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio — probability that the observed Sharpe is real,
    accounting for n_trials independent backtests considered.

    Returns a value in [0, 1]: probability that true Sharpe > 0.
    DSR > 0.95 → strong evidence of real edge.
    """
    r = _safe_returns(returns)
    if len(r) < 30:
        return 0.0

    sr = sharpe_ratio(r, periods_per_year)
    sr_per_period = sr / math.sqrt(periods_per_year)

    n = len(r)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt())  # excess kurtosis

    # Expected max Sharpe under null hypothesis of n_trials random strategies
    # E[max SR] ≈ sqrt(2 * ln(n_trials)) for large n_trials
    if n_trials <= 1:
        sr_zero = 0.0
    else:
        gamma = 0.5772156649  # Euler-Mascheroni
        e_max = math.sqrt(2 * math.log(n_trials))
        sr_zero = e_max - gamma / e_max if e_max > 0 else 0.0

    # Variance of estimator
    denom = 1 - skew * sr_per_period + (kurt / 4) * sr_per_period**2
    if denom <= 0:
        return 0.0
    var_sr = denom / (n - 1)
    sigma_sr = math.sqrt(var_sr)

    if sigma_sr == 0:
        return 1.0 if sr > sr_zero else 0.0

    # Standardized statistic
    z = (sr_per_period - sr_zero) / sigma_sr
    # Probit (cumulative normal)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Max drawdown as a positive fraction (0.20 = 20% drop)."""
    eq = pd.Series(equity).dropna()
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(-dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / MaxDD — penalizes deep drawdowns more than Sharpe."""
    r = _safe_returns(returns)
    if len(r) < 2:
        return 0.0
    cagr = (1 + r.mean()) ** periods_per_year - 1
    equity = (1 + pd.Series(r)).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / mdd)


def profit_factor(trade_pnls: pd.Series | np.ndarray) -> float:
    """Sum of wins / Sum of |losses|. > 1.5 = robust strategy."""
    arr = np.asarray(trade_pnls, dtype=float)
    wins = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(trade_pnls: pd.Series | np.ndarray) -> float:
    arr = np.asarray(trade_pnls, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float((arr > 0).mean())


def summary_stats(
    returns: pd.Series,
    trade_pnls: pd.Series | None = None,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """One-shot dashboard of all key metrics."""
    equity = (1 + returns.fillna(0)).cumprod()
    cagr = float(equity.iloc[-1] ** (periods_per_year / len(returns)) - 1) if len(returns) > 0 else 0.0
    out = {
        "n_periods": len(returns),
        "cagr": cagr,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "deflated_sharpe": deflated_sharpe_ratio(returns, n_trials, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns, periods_per_year),
        "vol_annual": float(returns.std() * math.sqrt(periods_per_year)),
    }
    if trade_pnls is not None and len(trade_pnls) > 0:
        out["n_trades"] = int(len(trade_pnls))
        out["win_rate"] = win_rate(trade_pnls)
        out["profit_factor"] = profit_factor(trade_pnls)
        out["avg_win"] = float(trade_pnls[trade_pnls > 0].mean()) if (trade_pnls > 0).any() else 0.0
        out["avg_loss"] = float(trade_pnls[trade_pnls < 0].mean()) if (trade_pnls < 0).any() else 0.0
    return out
