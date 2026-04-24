"""
TradeFlow — Sentiment Analyzer
Collects market sentiment from free sources and normalizes to 0-1 score.
  0  = extreme fear / strong sell signal
  1  = extreme greed / strong buy signal

Sources (all free, no API key):
  1. Fear & Greed Index (alternative.me)
  2. News sentiment via yfinance + TextBlob NLP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FNG_API_URL = "https://api.alternative.me/fng/?limit=1"
FNG_CACHE_MAX_AGE_SECONDS = 1800  # 30 min


@dataclass
class SentimentScores:
    """Normalized sentiment scores, all on 0-1 scale (0=fear/sell, 1=greed/buy)."""
    fear_greed: float       # From alternative.me Fear & Greed Index
    news_sentiment: float   # From news headlines NLP analysis
    composite: float        # Weighted average: 50% news + 50% F&G

    def to_dict(self) -> dict:
        return {
            "fear_greed": round(self.fear_greed, 3),
            "news_sentiment": round(self.news_sentiment, 3),
            "composite": round(self.composite, 3),
        }


def fetch_fear_greed_index() -> Optional[float]:
    """
    Fetch the current Crypto Fear & Greed Index from alternative.me.
    Returns value 0-100, normalized to 0-1. Returns None on failure.
    """
    try:
        resp = requests.get(FNG_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        value = int(data["data"][0]["value"])
        # Normalize 0-100 → 0-1
        return value / 100.0
    except Exception as exc:
        logger.warning("Fear & Greed Index fetch failed: %s", exc)
        return None


def analyze_news_sentiment(symbol: str) -> Optional[float]:
    """
    Fetch recent news for a symbol via yfinance and score sentiment with TextBlob.
    Returns a 0-1 score where 0=very negative, 1=very positive.
    """
    try:
        import yfinance as yf
        from textblob import TextBlob

        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            logger.debug("No news found for %s", symbol)
            return None

        scores = []
        for article in news[:15]:
            # yfinance >= 0.2.54 nests title inside 'content'
            title = ""
            if "content" in article and isinstance(article["content"], dict):
                title = article["content"].get("title", "")
            if not title:
                title = article.get("title", "")
            if not title:
                continue
            blob = TextBlob(title)
            polarity = blob.sentiment.polarity  # -1.0 to 1.0
            scores.append(polarity)

        if not scores:
            return None

        avg_polarity = sum(scores) / len(scores)
        # Normalize -1..1 → 0..1
        normalized = (avg_polarity + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    except ImportError:
        logger.warning("textblob not installed — skipping news sentiment")
        return None
    except Exception as exc:
        logger.warning("News sentiment analysis failed for %s: %s", symbol, exc)
        return None


def get_sentiment_scores(symbol: str) -> SentimentScores:
    """
    Compute all sentiment scores for a given symbol.
    Falls back to 0.5 (neutral) when a source is unavailable.
    """
    fng = fetch_fear_greed_index()
    news = analyze_news_sentiment(symbol)

    fng_score = fng if fng is not None else 0.5
    news_score = news if news is not None else 0.5

    composite = 0.5 * news_score + 0.5 * fng_score

    return SentimentScores(
        fear_greed=fng_score,
        news_sentiment=news_score,
        composite=composite,
    )