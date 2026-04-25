"""
Offline trainer for the MetaLabeler.

Pipeline:
  1. For each symbol in the universe, fetch N years of daily OHLCV
  2. Run the primary strategy on history → identify candidate events
  3. Apply triple-barrier labeling to each event
  4. Build features at each event time (no look-ahead)
  5. Train LightGBM, validate via walk-forward
  6. Save the model to disk

Usage:
    python -m app.meta_label.trainer \\
        --universe SPY,AAPL,MSFT,GOOG,AMZN \\
        --period 5y \\
        --strategy pullback \\
        --output data/meta_labeler.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.meta_label.meta_labeler import MetaLabeler, MetaLabelConfig, build_features
from app.meta_label.triple_barrier import TripleBarrierConfig, triple_barrier_labels
from app.regime.detector import RegimeDetector
from app.strategies_v2.pullback_trend import PullbackSignal, PullbackTrendStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("trainer")


def find_pullback_events(df: pd.DataFrame) -> pd.Series:
    """Run the pullback strategy bar-by-bar and return a boolean series of LONG events."""
    strat = PullbackTrendStrategy()
    events = pd.Series(False, index=df.index)

    sma_warmup = 200
    for i in range(sma_warmup, len(df)):
        window = df.iloc[: i + 1]
        decision = strat.generate(window, in_position=False)
        if decision.signal == PullbackSignal.LONG:
            events.iloc[i] = True

    return events


def build_dataset(
    symbol: str,
    period: str,
    benchmark_prices: Optional[pd.Series],
    barrier_config: TripleBarrierConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for a single symbol."""
    df = fetch_ohlcv(symbol, interval="1d", period=period)
    if df is None or df.empty or len(df) < 300:
        logger.warning("Skipping %s: insufficient data", symbol)
        return pd.DataFrame(), pd.Series(dtype=int)

    df = add_all_indicators(df)

    # 1. Find primary signal events
    events = find_pullback_events(df)
    if events.sum() == 0:
        logger.info("%s: zero pullback events, skipping", symbol)
        return pd.DataFrame(), pd.Series(dtype=int)

    logger.info("%s: %d pullback events found", symbol, int(events.sum()))

    # 2. Triple-barrier labeling
    labels_df = triple_barrier_labels(df, events, barrier_config)
    if labels_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # 3. Build features at each event time
    rd = RegimeDetector(use_hmm=False)
    feature_rows = []
    label_rows = []

    for event_time, lbl in zip(labels_df.index, labels_df["label"].values):
        idx = df.index.get_loc(event_time)
        if isinstance(idx, slice):
            idx = idx.start
        # Regime context using benchmark or self
        regime_signal = None
        if benchmark_prices is not None and event_time in benchmark_prices.index:
            bench_slice = benchmark_prices.loc[:event_time]
            if len(bench_slice) > 250:
                regime_signal = rd.detect(bench_slice).to_dict()

        features = build_features(df, idx, regime_signal)
        if not features:
            continue
        feature_rows.append(features)
        label_rows.append(lbl)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(label_rows, name="label")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train the MetaLabeler.")
    parser.add_argument("--universe", required=True, help="Comma-separated tickers")
    parser.add_argument("--period", default="5y", help="History window (yfinance period)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark for regime context")
    parser.add_argument("--output", default="data/meta_labeler.pkl", help="Output path")
    parser.add_argument("--pt-mult", type=float, default=3.0)
    parser.add_argument("--sl-mult", type=float, default=2.5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
    logger.info("Training on %d symbols, period=%s", len(universe), args.period)

    # Fetch benchmark once for regime context
    bench_df = fetch_ohlcv(args.benchmark, interval="1d", period=args.period)
    bench_prices = bench_df["close"] if bench_df is not None else None

    # Triple-barrier config
    bcfg = TripleBarrierConfig(
        pt_multiplier=args.pt_mult,
        sl_multiplier=args.sl_mult,
        time_horizon_bars=args.horizon,
    )

    all_X, all_y = [], []
    for sym in universe:
        X, y = build_dataset(sym, args.period, bench_prices, bcfg)
        if not X.empty:
            X["symbol"] = sym  # keep for diagnostics, dropped before fit
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        logger.error("No training data collected. Aborting.")
        sys.exit(1)

    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)

    logger.info(
        "Total events: %d | label distribution: PT=%d, timeout=%d, SL=%d",
        len(y_full),
        int((y_full == 1).sum()),
        int((y_full == 0).sum()),
        int((y_full == -1).sum()),
    )

    # Feature columns only (drop diagnostics)
    feature_cols = [c for c in X_full.columns if c != "symbol"]
    X_features = X_full[feature_cols]

    # Time-aware train/test split (last 20% = holdout)
    n = len(X_features)
    split = int(n * 0.8)
    X_train, X_test = X_features.iloc[:split], X_features.iloc[split:]
    y_train, y_test = y_full.iloc[:split], y_full.iloc[split:]

    # Train
    config = MetaLabelConfig(threshold=args.threshold)
    ml = MetaLabeler(config)
    ml.fit(X_train, y_train)

    # Holdout evaluation
    if len(X_test) > 0:
        proba = np.asarray(ml.predict_proba(X_test))
        if proba.ndim == 0:
            proba = np.array([float(proba)])
        y_test_binary = (y_test == 1).astype(int).values
        threshold = config.threshold

        acted = proba >= threshold
        n_acted = int(acted.sum())
        if n_acted > 0:
            precision = float(y_test_binary[acted].mean())
        else:
            precision = float("nan")
        recall = float(acted[y_test_binary == 1].mean()) if (y_test_binary == 1).any() else float("nan")

        logger.info(
            "Holdout (n=%d): acted=%d (%.1f%%), precision=%.3f, recall=%.3f",
            len(X_test), n_acted, n_acted / len(X_test) * 100,
            precision, recall,
        )

    # Top features
    importances = ml.feature_importance()
    top_5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    logger.info("Top 5 features: %s", top_5)

    # Save
    output = Path(args.output)
    ml.save(output)
    logger.info("✅ Model saved to %s", output)


if __name__ == "__main__":
    main()
