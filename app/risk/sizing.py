"""Position sizing — risk-based, not cash-based."""
from __future__ import annotations


def position_size_by_risk(
    capital: float,
    entry: float,
    stop: float,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.15,
) -> float:
    """
    The professional standard: risk X% of capital per trade.

    If the trade hits its stop, we lose exactly `risk_pct` of capital.
    Hard cap at `max_position_pct` to avoid oversized positions when stop is tight.

    Args:
        capital: current total portfolio value
        entry: intended entry price
        stop: intended stop-loss price
        risk_pct: fraction of capital to risk per trade (default 1%)
        max_position_pct: hard cap on single position as % of capital

    Returns:
        Number of shares to buy (float, may be fractional).
    """
    if capital <= 0 or entry <= 0:
        return 0.0

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0.0

    dollars_at_risk = capital * risk_pct
    shares_by_risk = dollars_at_risk / risk_per_share

    max_shares_by_cap = (capital * max_position_pct) / entry

    return min(shares_by_risk, max_shares_by_cap)


def kelly_fractional(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly — safer than full Kelly.

    Full Kelly = W - (1-W)/R where W=win rate, R=win/loss ratio
    Fractional Kelly (0.25-0.5) captures ~75% of growth with ~50% less drawdown.

    Returns the optimal fraction of capital to risk (0-1).
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0

    full_kelly = win_rate - (1 - win_rate) / win_loss_ratio
    if full_kelly <= 0:
        return 0.0  # no edge → don't trade

    return min(full_kelly * fraction, 0.25)  # cap at 25% even if Kelly says more
