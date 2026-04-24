"""
TradeFlow — OHLCV Data Fetcher
Fetches price data from yfinance with SQLite caching layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.exc import SQLAlchemyError

from app.database.models import PriceCache
from app.database.session import get_session, init_database

logger = logging.getLogger(__name__)

# Maximum cache age before refreshing from API
CACHE_MAX_AGE_HOURS: int = 4

# Supported intervals and their maximum lookback periods
SUPPORTED_INTERVALS: dict[str, str] = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "1d": "max",
}


def _cap_period(interval: str, period: str) -> str:
    """Cap the lookback period to the maximum allowed for the given interval.

    yfinance has tighter limits on shorter intervals (e.g. 15m max 60d).
    """
    max_period = SUPPORTED_INTERVALS.get(interval)
    if max_period is None or max_period == "max":
        return period

    # Normalize both to days for comparison
    period_days = _parse_period_to_days(period)
    max_days = _parse_period_to_days(max_period)
    if period_days is not None and max_days is not None and period_days > max_days:
        return max_period
    return period


def _parse_period_to_days(period: str) -> int | None:
    """Parse a yfinance period string to approximate number of days."""
    period = period.strip().lower()
    if period == "max":
        return None
    import re
    m = re.match(r"(\d+)([dmoys])", period)
    if not m:
        return None
    val = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return val
    elif unit == "m":
        return val * 30
    elif unit == "o":
        return val * 30  # month alias
    elif unit == "y":
        return val * 365
    return None


def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    period: str = "3mo",
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price data for a given symbol using yfinance, with SQLite caching.

    Args:
        symbol: Asset ticker symbol (e.g., 'AAPL', 'MC.PA').
        interval: Bar interval — one of '1m', '5m', '15m', '30m', '1h', '1d'.
        period: Lookback period string accepted by yfinance (e.g., '3mo', '1y').
        use_cache: If True, attempts to serve data from SQLite cache first.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by datetime,
        or None if the fetch fails.

    Raises:
        ValueError: If the interval is not supported.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Interval '{interval}' not supported. "
            f"Choose from: {list(SUPPORTED_INTERVALS.keys())}"
        )

    # Cap period to the maximum allowed for this interval
    period = _cap_period(interval, period)

    init_database()

    if use_cache:
        cached_df = _load_from_cache(symbol, interval)
        if cached_df is not None and not cached_df.empty:
            logger.info("Cache hit for %s [%s]", symbol, interval)
            return cached_df

    logger.info("Fetching %s [%s / %s] from yfinance…", symbol, interval, period)
    return _fetch_from_yfinance(symbol, interval, period)


def _fetch_from_yfinance(
    symbol: str,
    interval: str,
    period: str,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data from yfinance and persist it to the SQLite cache.

    Args:
        symbol: Asset ticker symbol.
        interval: Bar interval.
        period: Lookback period.

    Returns:
        Cleaned DataFrame, or None on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        # auto_adjust=True is default in yfinance >= 0.2.x — prices are split/dividend adjusted
        raw_df: pd.DataFrame = ticker.history(period=period, interval=interval, auto_adjust=True)

        if raw_df is None or raw_df.empty:
            logger.warning("yfinance returned empty data for %s [%s]", symbol, interval)
            return None

        # Normalize column names to lowercase (yfinance < 0.2.54 uses PascalCase,
        # newer versions may use lowercase — handle both defensively)
        raw_df.columns = [str(c).lower() for c in raw_df.columns]

        # Select only OHLCV columns — ignore Dividends, Stock Splits, Capital Gains, etc.
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in raw_df.columns]
        if missing:
            logger.error(
                "yfinance response missing columns %s for %s. Available: %s",
                missing,
                symbol,
                list(raw_df.columns),
            )
            return None

        df = raw_df[required_cols].copy()

        # Remove timezone info for SQLite compatibility
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # Cast all numeric columns to float
        df = df.astype(float)
        
        # Soft cleaning: forward fill close price, then infer others if missing
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)

        # Drop only rows where price is fundamentally missing/invalid
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        if df.empty:
            logger.warning("All bars filtered out for %s [%s] (no valid close prices)", symbol, interval)
            return None

        _save_to_cache(symbol, interval, df)
        logger.info("Fetched %d bars for %s [%s]", len(df), symbol, interval)
        return df

    except Exception as exc:
        logger.error("Failed to fetch %s from yfinance: %s", symbol, exc)
        return None


def _load_from_cache(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from SQLite cache if it is recent enough.

    Args:
        symbol: Asset ticker symbol.
        interval: Bar interval.

    Returns:
        DataFrame from cache, or None if stale/missing.
    """
    session = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=CACHE_MAX_AGE_HOURS)

        rows = (
            session.query(PriceCache)
            .filter(
                PriceCache.symbol == symbol,
                PriceCache.interval == interval,
                PriceCache.timestamp >= cutoff,
            )
            .order_by(PriceCache.timestamp.asc())
            .all()
        )

        if not rows:
            return None

        records = [
            {
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in rows
        ]

        df = pd.DataFrame(records)
        df = df.set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index)
        return df

    except SQLAlchemyError as exc:
        logger.error("Cache read error for %s: %s", symbol, exc)
        return None
    finally:
        session.close()


def _save_to_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    """
    Persist OHLCV DataFrame to SQLite PriceCache table.
    Existing entries for this symbol/interval are replaced.

    Args:
        symbol: Asset ticker symbol.
        interval: Bar interval.
        df: OHLCV DataFrame indexed by datetime.
    """
    session = get_session()
    try:
        # Delete stale entries for this symbol/interval
        session.query(PriceCache).filter(
            PriceCache.symbol == symbol,
            PriceCache.interval == interval,
        ).delete()

        cache_rows = [
            PriceCache(
                symbol=symbol,
                interval=interval,
                timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for idx, row in df.iterrows()
        ]

        session.bulk_save_objects(cache_rows)
        session.commit()
        logger.debug("Cached %d rows for %s [%s]", len(cache_rows), symbol, interval)

    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Failed to write cache for %s: %s", symbol, exc)
    finally:
        session.close()


def get_available_symbols() -> list[str]:
    """
    Return the list of symbols currently stored in the price cache.

    Returns:
        Sorted list of unique symbol strings.
    """
    session = get_session()
    try:
        rows = session.query(PriceCache.symbol).distinct().all()
        return sorted([row[0] for row in rows])
    except SQLAlchemyError as exc:
        logger.error("Could not query cached symbols: %s", exc)
        return []
    finally:
        session.close()
