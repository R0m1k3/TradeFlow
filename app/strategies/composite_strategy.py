"""
TradeFlow — Composite Strategy
Uses the 0-1 composite scoring system to generate BUY/SELL/HOLD signals.
  score > 0.7 → BUY
  score < 0.3 → SELL
  otherwise   → HOLD
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.analysis.composite import compute_composite_score, CompositeScore
from app.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

BUY_THRESHOLD = 0.7
SELL_THRESHOLD = 0.3


class CompositeStrategy(BaseStrategy):
    """
    Multi-factor composite strategy combining technical + sentiment + momentum.

    Score 0-1 where:
      > 0.7 = BUY (strong conviction)
      < 0.3 = SELL (strong conviction)
      else  = HOLD
    """

    def __init__(
        self,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
    ) -> None:
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._last_score: CompositeScore | None = None

    @property
    def name(self) -> str:
        return f"Composite [>{self._buy_threshold}/<{self._sell_threshold}]"

    @property
    def last_score(self) -> CompositeScore | None:
        return self._last_score

    def explain(self) -> str:
        """Return a plain French explanation of the last computed score."""
        if self._last_score is None:
            return ""
        s = self._last_score.combined
        if s >= 0.85:
            return "Les indicateurs sont tres favorables, c'est le moment d'acheter"
        if s >= 0.7:
            return "Le marche montre des signaux positifs, on peut acheter"
        if s >= 0.55:
            return "Legerement positif, mais pas assez pour acheter en confiance"
        if s >= 0.45:
            return "Pas de signal clair, on attend de voir"
        if s >= 0.3:
            return "Le marche hesite, les signaux sont legerement negatifs"
        if s >= 0.15:
            return "Les indicateurs sont defavorables, mieux vaut vendre"
        return "Le marche est tres pessimiste, signal fort de vente"

    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> tuple[Signal, str]:
        symbol = df.attrs.get("symbol", "???")
        score = compute_composite_score(df, symbol)
        self._last_score = score

        s = score.combined

        if s > self._buy_threshold:
            strength = "FORT" if s > 0.85 else "MODERE"
            reason = (
                f"ACHAT {strength} (score={s:.2f}) — "
                f"Tech={score.technical:.2f} Sentiment={score.sentiment:.2f} Momentum={score.momentum:.2f} | "
                f"RSI={score.rsi_score:.2f} MACD={score.macd_score:.2f} BB={score.bollinger_score:.2f} SMA={score.sma_score:.2f}"
            )
            return Signal.BUY, reason

        if s < self._sell_threshold:
            strength = "FORT" if s < 0.15 else "MODERE"
            reason = (
                f"VENTE {strength} (score={s:.2f}) — "
                f"Tech={score.technical:.2f} Sentiment={score.sentiment:.2f} Momentum={score.momentum:.2f} | "
                f"RSI={score.rsi_score:.2f} MACD={score.macd_score:.2f} BB={score.bollinger_score:.2f} SMA={score.sma_score:.2f}"
            )
            return Signal.SELL, reason

        reason = (
            f"NEUTRE (score={s:.2f}) — "
            f"Tech={score.technical:.2f} Sentiment={score.sentiment:.2f} Momentum={score.momentum:.2f}"
        )
        return Signal.HOLD, reason

    def get_params(self) -> dict[str, Any]:
        return {
            "buy_threshold": self._buy_threshold,
            "sell_threshold": self._sell_threshold,
        }