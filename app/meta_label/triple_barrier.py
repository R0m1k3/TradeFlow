"""
Triple-Barrier Labeling (Marcos López de Prado, 2018).

For each candidate trade event, apply three barriers simultaneously:
  - upper barrier: profit-taking level (entry + pt * ATR)
  - lower barrier: stop-loss level (entry - sl * ATR)
  - vertical barrier: time horizon (max N bars)

Label:
   1  if upper hit first (trade would have been profitable)
  -1  if lower hit first (trade would have been stopped out)
   0  if time barrier hit first (indecisive)

This is the canonical supervised-learning target for meta-labeling:
"would this signal, if acted upon, have made money?"

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018), ch.3
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.risk.stops import compute_atr


@dataclass
class TripleBarrierConfig:
    pt_multiplier: float = 3.0   # profit take = 3 × ATR (RR 1.2:1 vs sl=2.5)
    sl_multiplier: float = 2.5   # stop loss   = 2.5 × ATR
    time_horizon_bars: int = 20  # vertical barrier in bars


def triple_barrier_labels(
    df: pd.DataFrame,
    events: pd.Series,
    config: TripleBarrierConfig | None = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Label each event in `events` using the triple-barrier method.

    Args:
        df: OHLCV DataFrame indexed by datetime
        events: Boolean series (same index as df) — True where a primary signal
                fired. Only these bars get labeled.
        config: barrier multipliers and horizon
        atr_period: ATR lookback

    Returns:
        DataFrame with columns ['entry', 'pt', 'sl', 'first_hit', 'label',
                                'bars_to_hit', 'ret']
        indexed by the event timestamps.
    """
    if config is None:
        config = TripleBarrierConfig()

    atr = compute_atr(df, atr_period)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    records = []
    event_indices = np.where(events.values)[0]

    for idx in event_indices:
        if idx + 1 >= len(df):
            continue

        entry = float(close.iloc[idx])
        a = float(atr.iloc[idx])
        if pd.isna(a) or a <= 0:
            continue

        pt = entry + config.pt_multiplier * a
        sl = entry - config.sl_multiplier * a
        horizon_end = min(idx + config.time_horizon_bars, len(df) - 1)

        label = 0
        first_hit = "timeout"
        bars_to_hit = horizon_end - idx
        exit_price = float(close.iloc[horizon_end])

        # Walk forward to find first barrier hit
        for j in range(idx + 1, horizon_end + 1):
            h = float(high.iloc[j])
            l = float(low.iloc[j])
            if h >= pt:
                label = 1
                first_hit = "pt"
                bars_to_hit = j - idx
                exit_price = pt
                break
            if l <= sl:
                label = -1
                first_hit = "sl"
                bars_to_hit = j - idx
                exit_price = sl
                break

        records.append({
            "event_time": df.index[idx],
            "entry": entry,
            "pt": pt,
            "sl": sl,
            "atr": a,
            "first_hit": first_hit,
            "label": label,
            "bars_to_hit": bars_to_hit,
            "exit_price": exit_price,
            "ret": (exit_price - entry) / entry,
        })

    if not records:
        return pd.DataFrame(columns=["entry", "pt", "sl", "first_hit", "label", "bars_to_hit", "ret"])

    result = pd.DataFrame(records).set_index("event_time")
    return result
