"""
PullbackTrend — tactical mean-reversion ONLY inside confirmed uptrend.

This is the "clean" mean-reversion: we never buy a falling knife. We only
buy a pullback that happens inside an established uptrend, with confirmation
of a reversal.

Conditions for LONG entry:
  1. Trend confirmed : close > SMA200 AND SMA50 > SMA200
  2. Pullback active : close < SMA20
  3. Oversold        : RSI(14) < 35
  4. Rebound started : current close > previous close
  5. Volume OK       : 20-day avg volume > min_liquidity

Exit:
  - Price crosses back above SMA20 (take profit on mean-reversion)
  - OR ATR trailing stop hit (managed by RiskManager)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from app.risk.stops import compute_atr

logger = logging.getLogger(__name__)


class PullbackSignal(str, Enum):
    LONG = "long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"


@dataclass
class PullbackDecision:
    signal: PullbackSignal
    atr: float
    entry: float
    reason: str
    diagnostics: dict


class PullbackTrendStrategy:
    def __init__(
        self,
        sma_fast: int = 20,
        sma_mid: int = 50,
        sma_slow: int = 200,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        atr_period: int = 14,
        min_avg_volume: float = 500_000.0,
    ) -> None:
        self.sma_fast = sma_fast
        self.sma_mid = sma_mid
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.min_avg_volume = min_avg_volume

    def generate(self, df: pd.DataFrame, in_position: bool = False) -> PullbackDecision:
        """
        Args:
            df: OHLCV with indicators already computed (sma_*, rsi_*, volume)
            in_position: True if we already hold this symbol

        Returns:
            PullbackDecision with signal + ATR (for stop sizing).
        """
        if len(df) < self.sma_slow + 5:
            return PullbackDecision(
                signal=PullbackSignal.HOLD,
                atr=0.0,
                entry=0.0,
                reason="insufficient history",
                diagnostics={},
            )

        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        # Trend check
        sma_fast = df.get(f"sma_{self.sma_fast}")
        sma_mid = df.get(f"sma_{self.sma_mid}")
        sma_slow = df.get(f"sma_{self.sma_slow}")

        if sma_fast is None or sma_mid is None or sma_slow is None:
            return PullbackDecision(
                signal=PullbackSignal.HOLD,
                atr=0.0,
                entry=close,
                reason="missing SMAs (run add_all_indicators first)",
                diagnostics={},
            )

        sma20, sma50, sma200 = sma_fast.iloc[-1], sma_mid.iloc[-1], sma_slow.iloc[-1]
        rsi = df.get(f"rsi_{self.rsi_period}")
        rsi_now = rsi.iloc[-1] if rsi is not None else 50

        # Volume
        avg_vol = df["volume"].iloc[-20:].mean() if "volume" in df.columns else 0
        atr = compute_atr(df, self.atr_period).iloc[-1]

        diag = {
            "close": float(close),
            "sma20": float(sma20),
            "sma50": float(sma50),
            "sma200": float(sma200),
            "rsi": float(rsi_now),
            "atr": float(atr) if not pd.isna(atr) else 0,
            "avg_volume_20d": float(avg_vol),
        }

        # Exit logic (if in position)
        if in_position:
            # Mean-reversion completed: price back above SMA20
            if close > sma20 and prev_close <= sma20:
                return PullbackDecision(
                    signal=PullbackSignal.EXIT_LONG,
                    atr=atr,
                    entry=close,
                    reason="pullback reversion complete (close > SMA20)",
                    diagnostics=diag,
                )
            return PullbackDecision(
                signal=PullbackSignal.HOLD,
                atr=atr,
                entry=close,
                reason="hold in position",
                diagnostics=diag,
            )

        # Entry logic — all conditions must pass
        checks = {
            "trend_confirmed": close > sma200 and sma50 > sma200,
            "pullback_active": close < sma20,
            "oversold": rsi_now < self.rsi_oversold,
            "rebound_started": close > prev_close,
            "liquid": avg_vol >= self.min_avg_volume,
        }
        all_pass = all(checks.values())

        if all_pass:
            return PullbackDecision(
                signal=PullbackSignal.LONG,
                atr=float(atr) if not pd.isna(atr) else 0.0,
                entry=close,
                reason=(
                    f"Pullback in uptrend: close={close:.2f} < SMA20={sma20:.2f}, "
                    f"RSI={rsi_now:.1f} < {self.rsi_oversold}, rebound confirmed"
                ),
                diagnostics={**diag, **checks},
            )

        failed = [k for k, v in checks.items() if not v]
        return PullbackDecision(
            signal=PullbackSignal.HOLD,
            atr=float(atr) if not pd.isna(atr) else 0.0,
            entry=close,
            reason=f"conditions failed: {', '.join(failed)}",
            diagnostics={**diag, **checks},
        )
