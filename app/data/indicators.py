"""
TradeFlow — Technical Indicators
Calculates SMA, RSI, MACD, and Bollinger Bands using pure pandas/numpy.
Replaces the pandas-ta dependency to ensure robust cross-platform Docker builds.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_sma(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Add Simple Moving Average (SMA) columns to the DataFrame.
    """
    if windows is None:
        windows = [20, 50, 200]

    result = df.copy()
    for window in windows:
        col_name = f"sma_{window}"
        result[col_name] = result["close"].rolling(window=window, min_periods=1).mean()
        # Set first window-1 values to NaN to match pandas-ta behavior
        result.loc[result.index[: window - 1], col_name] = np.nan
        logger.debug("Added SMA(%d)", window)

    return result


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) column to the DataFrame.
    """
    result = df.copy()
    
    delta = result["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate exponential moving average of gains and losses
    # Using adjust=False to match standard RSI smoothing (Wilder's method)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when loss is 0)
    rsi = rsi.where(avg_loss != 0, 100)
    
    col_name = f"rsi_{period}"
    result[col_name] = rsi
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
    Uses pandas ewm to calculate the exponential moving averages.
    """
    result = df.copy()
    
    # Calculate Fast and Slow EMAs
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    result[f"MACD_{fast}_{slow}_{signal}"] = macd_line
    result[f"MACDs_{fast}_{slow}_{signal}"] = signal_line
    result[f"MACDh_{fast}_{slow}_{signal}"] = histogram
    
    logger.debug("Added MACD(%d,%d,%d)", fast, slow, signal)
    return result


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """
    Add Bollinger Bands columns to the DataFrame.
    """
    result = df.copy()
    
    # Calculate Middle Band (SMA)
    middle_band = result["close"].rolling(window=window).mean()
    
    # Calculate Standard Deviation
    rolling_std = result["close"].rolling(window=window).std(ddof=0)
    
    # Calculate Upper and Lower Bands
    upper_band = middle_band + (rolling_std * std)
    lower_band = middle_band - (rolling_std * std)
    
    # Calculate Bandwidth and Percent B (for completeness with pandas-ta)
    bandwidth = (upper_band - lower_band) / middle_band * 100
    percent_b = (result["close"] - lower_band) / (upper_band - lower_band)
    
    result[f"BBL_{window}_{std}"] = lower_band
    result[f"BBM_{window}_{std}"] = middle_band
    result[f"BBU_{window}_{std}"] = upper_band
    result[f"BBB_{window}_{std}"] = bandwidth
    result[f"BBP_{window}_{std}"] = percent_b
    
    logger.debug("Added Bollinger Bands(%d, %.1f)", window, std)
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all standard indicators at once.
    """
    result = df.copy()
    result = add_sma(result, windows=[20, 50, 200])
    result = add_rsi(result, period=14)
    result = add_macd(result, fast=12, slow=26, signal=9)
    result = add_bollinger(result, window=20, std=2.0)
    return result
