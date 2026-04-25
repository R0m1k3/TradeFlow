"""Stop-loss calculations — ATR-based (volatility-adaptive)."""
from __future__ import annotations

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — volatility measure used for dynamic stops.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder's smoothing of TR over `period` bars.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (equivalent to EMA alpha=1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def atr_stop(entry: float, atr: float, multiplier: float = 2.5, side: str = "long") -> float:
    """
    Compute initial stop-loss price based on ATR.

    - 2.0x ATR: tight (swing trades)
    - 2.5x ATR: balanced (default, Van Tharp recommendation)
    - 3.0x ATR: loose (position trades, noisy assets)
    """
    if side == "long":
        return entry - multiplier * atr
    return entry + multiplier * atr


def trailing_stop_price(
    initial_stop: float,
    current_high: float,
    atr: float,
    multiplier: float = 2.5,
    side: str = "long",
) -> float:
    """
    Trailing stop — ratchets with the trade, never loosens.
    Locks in profits as price moves favorably.
    """
    if side == "long":
        candidate = current_high - multiplier * atr
        return max(initial_stop, candidate)
    candidate = current_high + multiplier * atr
    return min(initial_stop, candidate)


def stop_distance_pct(entry: float, stop: float) -> float:
    """Risk distance expressed as % of entry price."""
    return abs(entry - stop) / entry
