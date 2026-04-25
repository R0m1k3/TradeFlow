"""
Regime detector — tells the bot when to trade, when to stand aside.

Three layers stack:
  1. Trend filter (simple, robust)   : price > SMA200 = bullish
  2. Volatility level (realized + VIX): high vol = reduce exposure
  3. HMM regime (optional, data-driven): 2-state Gaussian HMM on returns

Decision rule:
  BULL_CALM  → full exposure
  BULL_VOL   → half exposure (position size ÷ 2)
  BEAR       → no new longs (close existing at stops)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL_CALM = "bull_calm"
    BULL_VOL = "bull_vol"
    BEAR = "bear"


@dataclass
class RegimeSignal:
    regime: Regime
    trend_up: bool              # price > SMA200
    realized_vol_annual: float  # annualized σ
    vol_percentile: float       # 0-1 vs history
    hmm_state: Optional[int] = None
    hmm_state_prob: Optional[float] = None
    exposure_multiplier: float = 1.0  # to be applied to position sizing

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "trend_up": self.trend_up,
            "realized_vol_annual": round(self.realized_vol_annual, 4),
            "vol_percentile": round(self.vol_percentile, 3),
            "hmm_state": self.hmm_state,
            "hmm_state_prob": round(self.hmm_state_prob, 3) if self.hmm_state_prob else None,
            "exposure_multiplier": self.exposure_multiplier,
        }


class RegimeDetector:
    """Detect market regime from a benchmark index (SPY for US, CAC for EU)."""

    def __init__(
        self,
        sma_period: int = 200,
        vol_lookback: int = 20,
        vol_history_window: int = 252 * 2,
        high_vol_percentile: float = 0.75,
        use_hmm: bool = True,
    ) -> None:
        self.sma_period = sma_period
        self.vol_lookback = vol_lookback
        self.vol_history_window = vol_history_window
        self.high_vol_percentile = high_vol_percentile
        self.use_hmm = use_hmm
        self._hmm = None
        self._hmm_fitted_len = 0

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, prices: pd.Series) -> RegimeSignal:
        """
        Detect the current regime from a price series (daily bars preferred).

        Args:
            prices: close prices of the benchmark (e.g., SPY), oldest first.
        """
        if len(prices) < max(self.sma_period, self.vol_history_window // 2):
            logger.warning("Insufficient data for regime detection (len=%d)", len(prices))
            return RegimeSignal(
                regime=Regime.BULL_CALM,
                trend_up=True,
                realized_vol_annual=0.0,
                vol_percentile=0.5,
                exposure_multiplier=0.5,  # conservative default
            )

        # Layer 1 — Trend filter
        sma = prices.rolling(self.sma_period, min_periods=self.sma_period // 2).mean()
        trend_up = bool(prices.iloc[-1] > sma.iloc[-1])

        # Layer 2 — Volatility
        returns = np.log(prices / prices.shift(1)).dropna()
        recent_vol = returns.iloc[-self.vol_lookback:].std()
        realized_vol_annual = recent_vol * np.sqrt(252)

        # Historical vol distribution for percentile
        hist_vol = returns.rolling(self.vol_lookback).std().iloc[-self.vol_history_window:]
        hist_vol = hist_vol.dropna()
        if len(hist_vol) > 10:
            vol_percentile = float((hist_vol < recent_vol).mean())
        else:
            vol_percentile = 0.5

        # Layer 3 — HMM (optional)
        hmm_state, hmm_prob = None, None
        if self.use_hmm:
            hmm_state, hmm_prob = self._hmm_regime(returns)

        # Combined decision
        regime, exposure = self._combine(trend_up, vol_percentile, hmm_state)

        return RegimeSignal(
            regime=regime,
            trend_up=trend_up,
            realized_vol_annual=float(realized_vol_annual),
            vol_percentile=vol_percentile,
            hmm_state=hmm_state,
            hmm_state_prob=hmm_prob,
            exposure_multiplier=exposure,
        )

    # ── HMM ───────────────────────────────────────────────────────────────────

    def _hmm_regime(self, returns: pd.Series) -> tuple[Optional[int], Optional[float]]:
        """
        Fit a 2-state Gaussian HMM on log returns.
        State 0: low-vol regime (usually bullish)
        State 1: high-vol regime (usually bearish / crisis)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.debug("hmmlearn not installed, skipping HMM")
            return None, None

        # Use last 2 years of daily returns
        X = returns.iloc[-500:].values.reshape(-1, 1)
        if len(X) < 100:
            return None, None

        try:
            # Refit only occasionally (expensive)
            if self._hmm is None or abs(len(X) - self._hmm_fitted_len) > 20:
                self._hmm = GaussianHMM(
                    n_components=2,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                )
                self._hmm.fit(X)
                self._hmm_fitted_len = len(X)

            # Identify which state is "high-vol" by comparing means/covariances
            covariances = self._hmm.covars_.flatten() if hasattr(self._hmm, "covars_") else None
            if covariances is not None:
                # State with higher variance = high-vol state
                high_vol_state = int(np.argmax(covariances))
            else:
                high_vol_state = 1

            # Current state probability
            probs = self._hmm.predict_proba(X[-1:])[0]
            current_state = int(np.argmax(probs))
            current_prob = float(probs[current_state])

            # Normalize: return 1 if in high-vol state, 0 otherwise
            normalized_state = 1 if current_state == high_vol_state else 0
            return normalized_state, current_prob
        except Exception as exc:
            logger.warning("HMM fit failed: %s", exc)
            return None, None

    # ── Decision logic ────────────────────────────────────────────────────────

    def _combine(
        self,
        trend_up: bool,
        vol_percentile: float,
        hmm_state: Optional[int],
    ) -> tuple[Regime, float]:
        """
        Decision matrix:
          trend_up=False              → BEAR (0× exposure)
          trend_up=True + vol high    → BULL_VOL (0.5× exposure)
          trend_up=True + hmm=high    → BULL_VOL
          trend_up=True + all calm    → BULL_CALM (1.0× exposure)
        """
        if not trend_up:
            return Regime.BEAR, 0.0

        # Trend is up
        high_vol_now = vol_percentile >= self.high_vol_percentile
        high_vol_hmm = hmm_state == 1

        if high_vol_now or high_vol_hmm:
            return Regime.BULL_VOL, 0.5

        return Regime.BULL_CALM, 1.0
