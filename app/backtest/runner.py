"""
Strategy backtest runner — walk-forward + CPCV evaluation.

For a given strategy and universe, computes:
  - Walk-forward equity curve, fold-by-fold metrics
  - CPCV Sharpe distribution + Probability of Backtest Overfitting (PBO)
  - Deflated Sharpe Ratio (DSR) using N=#paths as trial count
  - Summary table comparing strategy vs benchmark (buy & hold)

Usage:
    python -m app.backtest.runner \\
        --strategy pullback \\
        --universe SPY,AAPL,MSFT \\
        --period 5y \\
        --benchmark SPY
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.cpcv import CombinatorialPurgedCV
from app.backtest.metrics import (
    deflated_sharpe_ratio,
    sharpe_ratio,
    summary_stats,
)
from app.backtest.walkforward import WalkForward
from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.regime.detector import RegimeDetector
from app.risk.stops import compute_atr
from app.strategies_v2.pullback_trend import PullbackSignal, PullbackTrendStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backtest_runner")


# ── Strategy adapters: convert v2 strategies to (data) → returns ─────────────


def pullback_returns(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    risk_per_trade: float = 0.01,
    atr_mult_stop: float = 2.5,
    atr_mult_trail: float = 2.5,
    use_regime: bool = True,
    benchmark_prices: pd.Series | None = None,
) -> pd.Series:
    """
    Simulate the pullback strategy on a single symbol's history.
    Returns daily returns series indexed by df dates.
    """
    if len(df) < 220:
        return pd.Series(dtype=float)

    df = df.copy()
    if "sma_20" not in df.columns:
        df = add_all_indicators(df)

    strat = PullbackTrendStrategy()
    rd = RegimeDetector(use_hmm=False) if use_regime else None
    atr_series = compute_atr(df, 14)

    cash = initial_cash
    qty = 0.0
    stop = 0.0
    high_water = 0.0
    equity_series = []
    in_pos = False

    for i in range(200, len(df)):
        window = df.iloc[: i + 1]
        price = float(df["close"].iloc[i])
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

        # Mark-to-market equity
        equity = cash + qty * price
        equity_series.append((df.index[i], equity))

        # Manage existing position
        if in_pos:
            high_water = max(high_water, price)
            new_stop = max(stop, high_water - atr_mult_trail * atr)
            stop = new_stop

            decision = strat.generate(window, in_position=True)
            should_exit = price <= stop or decision.signal == PullbackSignal.EXIT_LONG
            if should_exit:
                cash += qty * price
                qty = 0.0
                in_pos = False
                stop = 0.0
                high_water = 0.0
            continue

        # Skip new entries if regime disallows
        if rd is not None and benchmark_prices is not None:
            bench_slice = benchmark_prices.loc[: df.index[i]]
            if len(bench_slice) > 220:
                regime_signal = rd.detect(bench_slice)
                if regime_signal.regime.value == "bear":
                    continue
                exposure = regime_signal.exposure_multiplier
            else:
                exposure = 1.0
        else:
            exposure = 1.0

        # Entry signal?
        decision = strat.generate(window, in_position=False)
        if decision.signal != PullbackSignal.LONG or atr <= 0:
            continue

        stop_price = price - atr_mult_stop * atr
        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            continue
        dollars_at_risk = equity * risk_per_trade * exposure
        shares = dollars_at_risk / risk_per_share
        max_shares = (equity * 0.15) / price
        shares = min(shares, max_shares, cash / price * 0.99)
        if shares <= 0:
            continue
        cost = shares * price
        cash -= cost
        qty = shares
        stop = stop_price
        high_water = price
        in_pos = True

    if not equity_series:
        return pd.Series(dtype=float)

    eq_df = pd.DataFrame(equity_series, columns=["date", "equity"]).set_index("date")
    eq = eq_df["equity"]
    returns = eq.pct_change().fillna(0)
    return returns


# ── Strategy wrapper for walk-forward / CPCV ─────────────────────────────────


@dataclass
class BacktestRunner:
    symbol: str
    period: str = "5y"
    benchmark: str = "SPY"

    def __post_init__(self):
        self.df = fetch_ohlcv(self.symbol, interval="1d", period=self.period)
        if self.df is None or self.df.empty:
            raise RuntimeError(f"No data for {self.symbol}")
        self.df = add_all_indicators(self.df)

        bench_df = fetch_ohlcv(self.benchmark, interval="1d", period=self.period)
        self.bench_prices = bench_df["close"] if bench_df is not None else None

    def run_walkforward(self, train_periods: int = 504, test_periods: int = 126):
        """Walk-forward: 2y train, 6mo test, slide forward."""
        wf = WalkForward(
            train_periods=train_periods,
            test_periods=test_periods,
            embargo=5,
            anchored=False,
        )

        def strategy_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
            # Pullback strategy doesn't require training (rules-based)
            # We just compute returns over the test window with full history available
            full_df = pd.concat([train_df, test_df]).sort_index()
            returns = pullback_returns(
                full_df,
                benchmark_prices=self.bench_prices,
            )
            return returns.loc[test_df.index[0]:test_df.index[-1]]

        return wf.run(self.df, strategy_fn)

    def run_cpcv(self, n_groups: int = 10, n_test_groups: int = 2):
        cpcv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)

        def strategy_fn(train_df, test_df):
            full_df = pd.concat([train_df, test_df]).sort_index()
            returns = pullback_returns(
                full_df,
                benchmark_prices=self.bench_prices,
            )
            return returns.loc[test_df.index[0]:test_df.index[-1]]

        return cpcv.run(self.df, strategy_fn)

    def run_full_history(self):
        """Run the strategy on the entire history (no folds)."""
        return pullback_returns(
            self.df,
            benchmark_prices=self.bench_prices,
        )

    def benchmark_returns(self) -> pd.Series:
        if self.bench_prices is None:
            return pd.Series(dtype=float)
        return self.bench_prices.pct_change().fillna(0)


# ── CLI ──────────────────────────────────────────────────────────────────────


def print_metrics_table(label: str, metrics: dict):
    print(f"\n── {label} ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:10.4f}")
        else:
            print(f"  {k:25s} {v}")


def main():
    parser = argparse.ArgumentParser(description="Backtest TradeFlow v2 strategies.")
    parser.add_argument("--strategy", default="pullback", choices=["pullback"])
    parser.add_argument("--universe", required=True, help="Comma-separated tickers")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--cpcv", action="store_true", help="Run CPCV (slow)")
    args = parser.parse_args()

    universe = [s.strip().upper() for s in args.universe.split(",")]

    aggregate_returns = []

    for sym in universe:
        print(f"\n{'='*60}\n  Backtest: {sym}\n{'='*60}")
        try:
            br = BacktestRunner(sym, period=args.period, benchmark=args.benchmark)
        except RuntimeError as exc:
            logger.warning(str(exc))
            continue

        # Full history
        ret_full = br.run_full_history()
        if not ret_full.empty:
            aggregate_returns.append(ret_full)
            stats = summary_stats(ret_full)
            print_metrics_table(f"{sym} — Full History", stats)

            bench_ret = br.benchmark_returns()
            if not bench_ret.empty:
                bench_stats = summary_stats(bench_ret.loc[ret_full.index[0]:ret_full.index[-1]])
                print_metrics_table(f"{args.benchmark} — Buy & Hold (same period)", bench_stats)

        # Walk-forward
        wf = br.run_walkforward()
        if wf.fold_returns:
            wf_returns = wf.combined_returns()
            stats = summary_stats(wf_returns)
            print_metrics_table(f"{sym} — Walk-Forward Combined", stats)
            wf_df = wf.to_dataframe()
            print("\n  Per-fold sharpes:")
            for _, row in wf_df.iterrows():
                print(f"    fold {int(row['fold'])}: sharpe={row['sharpe']:.2f}, maxDD={row['max_drawdown']:.2f}")

        # CPCV
        if args.cpcv:
            cpcv_result = br.run_cpcv()
            print("\n  CPCV results:")
            print(f"    n_paths        = {cpcv_result['n_paths']}")
            print(f"    mean Sharpe    = {cpcv_result['mean_sharpe']:.3f}")
            print(f"    median Sharpe  = {cpcv_result['median_sharpe']:.3f}")
            print(f"    std Sharpe     = {cpcv_result['std_sharpe']:.3f}")
            print(f"    PBO            = {cpcv_result['pbo']:.3f}")
            dsr = deflated_sharpe_ratio(
                ret_full, n_trials=cpcv_result['n_paths']
            ) if not ret_full.empty else 0
            print(f"    Deflated Sharpe (P{{SR>0}}) = {dsr:.3f}")

    # Aggregate (equal-weighted portfolio)
    if len(aggregate_returns) > 1:
        port = pd.concat(aggregate_returns, axis=1).fillna(0).mean(axis=1)
        port_stats = summary_stats(port)
        print_metrics_table(f"\n=== EQUAL-WEIGHTED PORTFOLIO ({len(aggregate_returns)} symbols) ===", port_stats)


if __name__ == "__main__":
    main()
