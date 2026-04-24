"""
TradeFlow — Composite Scoring System
Combines technical indicators + sentiment into a single 0-1 score.
  0  = strong sell / avoid
  1  = strong buy

Weights:
  - Technical: 50%
  - Sentiment (news + F&G): 30%
  - Momentum (volume trend): 20%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analysis.sentiment import get_sentiment_scores, SentimentScores

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    """Full breakdown of the composite 0-1 score."""
    technical: float      # Technical indicators normalized to 0-1
    sentiment: float      # Sentiment (news + F&G) normalized to 0-1
    momentum: float       # Volume/momentum normalized to 0-1
    combined: float       # Weighted composite: 50% tech + 30% sent + 20% mom

    # Sub-scores for display
    rsi_score: float
    macd_score: float
    bollinger_score: float
    sma_score: float
    fear_greed: float
    news_sentiment: float

    def to_dict(self) -> dict:
        return {
            "combined": round(self.combined, 3),
            "technical": round(self.technical, 3),
            "sentiment": round(self.sentiment, 3),
            "momentum": round(self.momentum, 3),
            "rsi_score": round(self.rsi_score, 3),
            "macd_score": round(self.macd_score, 3),
            "bollinger_score": round(self.bollinger_score, 3),
            "sma_score": round(self.sma_score, 3),
            "fear_greed": round(self.fear_greed, 3),
            "news_sentiment": round(self.news_sentiment, 3),
        }


def _score_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """RSI → 0-1: oversold (RSI<30) → 1.0 (buy), overbought (RSI>70) → 0.0 (sell)."""
    col = f"rsi_{period}"
    if col not in df.columns:
        return 0.5
    rsi = df[col].iloc[-1]
    if pd.isna(rsi):
        return 0.5
    # Invert: low RSI = buy signal → high score
    if rsi <= 30:
        return 1.0
    if rsi >= 70:
        return 0.0
    # Linear scale: RSI 30→70 maps to score 1.0→0.0
    return 1.0 - (rsi - 30) / 40


def _score_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> float:
    """MACD → 0-1: MACD above signal → bullish → 0.7+, below → bearish → 0.3-."""
    macd_col = f"MACD_{fast}_{slow}_{sig}"
    sig_col = f"MACDs_{fast}_{slow}_{sig}"
    hist_col = f"MACDh_{fast}_{slow}_{sig}"
    if macd_col not in df.columns or sig_col not in df.columns:
        return 0.5

    macd_now = df[macd_col].iloc[-1]
    sig_now = df[sig_col].iloc[-1]
    hist_now = df[hist_col].iloc[-1] if hist_col in df.columns else 0

    if pd.isna(macd_now) or pd.isna(sig_now):
        return 0.5

    diff = macd_now - sig_now
    # Normalize the histogram into 0-1 range
    price = df["close"].iloc[-1]
    if price <= 0:
        return 0.5
    normalized = diff / (price * 0.02)  # Scale relative to 2% of price
    normalized = max(-1.0, min(1.0, normalized))
    return (normalized + 1.0) / 2.0


def _score_bollinger(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> float:
    """Bollinger → 0-1: price near lower band → 0.8+ (buy), near upper → 0.2- (sell)."""
    lower_col = f"BBL_{window}_{std}"
    upper_col = f"BBU_{window}_{std}"
    pct_col = f"BBP_{window}_{std}"

    if pct_col in df.columns:
        bbp = df[pct_col].iloc[-1]
        if not pd.isna(bbp):
            # BBP is already 0-1 scale: 0=at lower band (buy), 1=at upper band (sell)
            # Invert for our scoring: low BBP → high buy score
            return 1.0 - bbp

    if lower_col not in df.columns or upper_col not in df.columns:
        return 0.5

    lower = df[lower_col].iloc[-1]
    upper = df[upper_col].iloc[-1]
    price = df["close"].iloc[-1]

    if pd.isna(lower) or pd.isna(upper) or upper == lower:
        return 0.5

    position = (price - lower) / (upper - lower)
    position = max(0.0, min(1.0, position))
    return 1.0 - position


def _score_sma(df: pd.DataFrame) -> float:
    """SMA trend → 0-1: SMA20 > SMA50 > SMA200 → strong uptrend → 0.8+."""
    sma20 = df.get("sma_20")
    sma50 = df.get("sma_50")
    sma200 = df.get("sma_200")
    price = df["close"].iloc[-1]

    score = 0.5

    if sma20 is not None and sma50 is not None:
        s20 = sma20.iloc[-1]
        s50 = sma50.iloc[-1]
        if not pd.isna(s20) and not pd.isna(s50):
            if s20 > s50:
                score += 0.15
            else:
                score -= 0.15

    if sma50 is not None and sma200 is not None:
        s50 = sma50.iloc[-1]
        s200 = sma200.iloc[-1]
        if not pd.isna(s50) and not pd.isna(s200):
            if s50 > s200:
                score += 0.15
            else:
                score -= 0.15

    if sma20 is not None:
        s20 = sma20.iloc[-1]
        if not pd.isna(s20):
            if price > s20:
                score += 0.1
            else:
                score -= 0.1

    return max(0.0, min(1.0, score))


def _score_momentum(df: pd.DataFrame) -> float:
    """Volume momentum → 0-1: increasing volume confirms trend direction."""
    if "volume" not in df.columns or len(df) < 20:
        return 0.5

    vol = df["volume"].iloc[-20:]
    if vol.sum() == 0:
        return 0.5

    recent_vol = vol.iloc[-5:].mean()
    older_vol = vol.iloc[:15].mean()

    if older_vol == 0:
        return 0.5

    vol_ratio = recent_vol / older_vol

    # Also check price direction to assign meaning
    price_now = df["close"].iloc[-1]
    price_prev = df["close"].iloc[-5] if len(df) >= 5 else price_now
    price_up = price_now > price_prev

    if vol_ratio > 1.5:
        # High volume: confirms the current direction
        return 0.75 if price_up else 0.25
    if vol_ratio > 1.0:
        return 0.65 if price_up else 0.35
    if vol_ratio < 0.5:
        # Low volume: weak conviction, neutral
        return 0.5
    return 0.5


def compute_composite_score(df: pd.DataFrame, symbol: str) -> CompositeScore:
    """
    Compute the full composite 0-1 score combining all signals.
    """
    rsi = _score_rsi(df)
    macd = _score_macd(df)
    boll = _score_bollinger(df)
    sma = _score_sma(df)
    momentum = _score_momentum(df)

    # Technical composite: average of sub-indicators
    technical = 0.30 * rsi + 0.25 * macd + 0.20 * boll + 0.25 * sma

    # Sentiment
    sentiment_data = get_sentiment_scores(symbol)
    sentiment = sentiment_data.composite

    # Final combined score
    combined = 0.50 * technical + 0.30 * sentiment + 0.20 * momentum

    return CompositeScore(
        technical=technical,
        sentiment=sentiment,
        momentum=momentum,
        combined=combined,
        rsi_score=rsi,
        macd_score=macd,
        bollinger_score=boll,
        sma_score=sma,
        fear_greed=sentiment_data.fear_greed,
        news_sentiment=sentiment_data.news_sentiment,
    )