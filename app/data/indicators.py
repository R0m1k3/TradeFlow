"""
TradeFlow — Technical Indicators
Adds SMA, RSI, MACD, and Bollinger Bands to OHLCV DataFrames using pandas_ta.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def add_sma(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Add Simple Moving Average (SMA) columns to the DataFrame.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        windows: List of SMA periods (default: [20, 50, 200]).

    Returns:
        DataFrame with added columns: sma_<window> for each window.
    """
    if windows is None:
        windows = [20, 50, 200]

    result = df.copy()
    for window in windows:
        col_name = f"sma_{window}"
        result[col_name] = ta.sma(result["close"], length=window)
        logger.debug("Added SMA(%d)", window)

    return result


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) column to the DataFrame.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: RSI lookback period (default: 14).

    Returns:
        DataFrame with added column: rsi_<period>.
    """
    result = df.copy()
    result[f"rsi_{period}"] = ta.rsi(result["close"], length=period)
    logger.debug("Added RSI(%d)", period)
    return result


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add MACD line, Signal line, and Histogram columns to the DataFrame.

    Column names follow pandas_ta convention:
        - MACD_{fast}_{slow}_{signal}        (MACD line)
        - MACDs_{fast}_{slow}_{signal}       (Signal line)
        - MACDh_{fast}_{slow}_{signal}       (Histogram)

    Args:
        df: OHLCV DataFrame with a 'close' column.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal EMA period (default: 9).

    Returns:
        DataFrame with added MACD columns.
    """
    result = df.copy()
    macd_df = ta.macd(result["close"], fast=fast, slow=slow, signal=signal)

    if macd_df is not None and not macd_df.empty:
        result = pd.concat([result, macd_df], axis=1)
        logger.debug("Added MACD(%d,%d,%d)", fast, slow, signal)
    else:
        logger.warning("MACD calculation returned empty result")

    return result


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """
    Add Bollinger Bands columns to the DataFrame.

    Column names follow pandas_ta convention:
        - BBL_{window}_{std}   (Lower band)
        - BBM_{window}_{std}   (Middle band / SMA)
        - BBU_{window}_{std}   (Upper band)
        - BBB_{window}_{std}   (Bandwidth)
        - BBP_{window}_{std}   (Percent B)

    Args:
        df: OHLCV DataFrame with a 'close' column.
        window: Rolling window period (default: 20).
        std: Standard deviation multiplier (default: 2.0).

    Returns:
        DataFrame with added Bollinger Bands columns.
    """
    result = df.copy()
    bb_df = ta.bbands(result["close"], length=window, std=std)

    if bb_df is not None and not bb_df.empty:
        result = pd.concat([result, bb_df], axis=1)
        logger.debug("Added Bollinger Bands(%d, %.1f)", window, std)
    else:
        logger.warning("Bollinger Bands calculation returned empty result")

    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all standard indicators at once.

    Applies SMA (20, 50, 200), RSI (14), MACD (12/26/9), and Bollinger Bands (20).

    Args:
        df: OHLCV DataFrame with a 'close' column.

    Returns:
        Fully enriched DataFrame with all indicator columns.
    """
    result = df.copy()
    result = add_sma(result, windows=[20, 50, 200])
    result = add_rsi(result, period=14)
    result = add_macd(result, fast=12, slow=26, signal=9)
    result = add_bollinger(result, window=20, std=2.0)
    return result
