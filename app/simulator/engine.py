"""
TradeFlow — Simulation Engine
Bar-by-bar backtesting engine that connects data, strategy, broker, and portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.database.models import Portfolio as PortfolioModel
from app.database.models import SimRun, Trade
from app.database.session import get_session, init_database
from app.simulator.broker import OrderSide, VirtualBroker
from app.simulator.portfolio import Portfolio
from app.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class SimResult:
    """
    Results of a completed simulation run.

    Attributes:
        sim_run_id: DB primary key of the SimRun record.
        strategy_name: Name of the strategy used.
        symbol: Traded asset symbol.
        interval: Bar interval.
        initial_capital: Starting cash.
        final_value: Final portfolio value.
        total_return_pct: Total return as percentage.
        sharpe_ratio: Annualized Sharpe ratio (0 if insufficient data).
        max_drawdown_pct: Maximum drawdown percentage.
        win_rate: Ratio of profitable trades.
        total_trades: Total number of executed trades.
        trades: List of trade record dicts.
        equity_curve: List of (timestamp, total_value) tuples.
    """
    sim_run_id: int
    strategy_name: str
    symbol: str
    interval: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)


class SimulationEngine:
    """
    Bar-by-bar backtesting engine.

    Orchestrates data fetch → indicator calculation → strategy signals
    → broker execution → portfolio update → DB persistence.

    Args:
        commission_rate: Broker commission per trade (fraction).
        slippage_rate: Broker slippage per trade (fraction).
        position_size_pct: Fraction of available cash to invest per trade (default: 0.95).
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size_pct: float = 0.95,
    ) -> None:
        self._commission_rate = commission_rate
        self._slippage_rate = slippage_rate
        self._position_size_pct = position_size_pct

    def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        interval: str = "1h",
        period: str = "3mo",
        initial_capital: float = 10_000.0,
        progress_callback: Optional[Callable[[float], None]] = None,
        save_to_db: bool = True,
    ) -> Optional[SimResult]:
        """
        Execute a full backtest simulation.

        Args:
            strategy: Instantiated strategy implementing BaseStrategy.
            symbol: Asset ticker symbol.
            interval: OHLCV bar interval.
            period: Historical data lookback period.
            initial_capital: Starting portfolio cash.
            progress_callback: Optional callable(pct: float) called each bar for UI updates.
            save_to_db: If True, persists trades and portfolio snapshots to SQLite.

        Returns:
            SimResult with all metrics and trade history, or None on failure.
        """
        init_database()

        logger.info(
            "Starting simulation: %s on %s [%s / %s] with %.2f capital",
            strategy.name,
            symbol,
            interval,
            period,
            initial_capital,
        )

        # ── 1. Fetch & enrich data ───────────────────────────────────────────
        df = fetch_ohlcv(symbol, interval=interval, period=period)
        if df is None or df.empty:
            logger.error("No data available for %s — simulation aborted", symbol)
            return None

        df = add_all_indicators(df)
        df = df.reset_index()  # Move DatetimeIndex to 'timestamp' column

        # Rename index column (yfinance may call it 'Datetime' or 'Date')
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        elif df.columns[0] != "timestamp":
            df = df.rename(columns={df.columns[0]: "timestamp"})

        start_date = str(df["timestamp"].iloc[0].date())
        end_date = str(df["timestamp"].iloc[-1].date())
        total_bars = len(df)

        # ── 2. Initialize broker, portfolio, and SimRun record ────────────────
        broker = VirtualBroker(
            commission_rate=self._commission_rate,
            slippage_rate=self._slippage_rate,
        )
        portfolio = Portfolio(initial_capital=initial_capital)

        if save_to_db:
            sim_run = self._create_sim_run(
                strategy=strategy,
                symbol=symbol,
                interval=interval,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
            )
            if sim_run is None:
                return None
            sim_run_id = sim_run.id
        else:
            sim_run_id = -1

        executed_trades: list[Trade] = []
        equity_curve: list[tuple[datetime, float]] = []

        # ── 3. Bar-by-bar simulation loop ────────────────────────────────────
        for idx in range(total_bars):
            bar = df.iloc[idx]
            current_price: float = float(bar["close"])
            current_time: datetime = bar["timestamp"].to_pydatetime() if hasattr(
                bar["timestamp"], "to_pydatetime"
            ) else bar["timestamp"]

            # Snapshot equity at every bar
            portfolio.take_snapshot(current_time, {symbol: current_price})
            equity_curve.append((current_time, portfolio.get_total_value({symbol: current_price})))

            # Progress callback for UI
            if progress_callback is not None:
                progress_callback((idx + 1) / total_bars)

            # Generate signal from strategy
            signal = strategy.generate_signal(df, idx)

            position = portfolio.get_position(symbol)

            if signal == Signal.BUY and position is None:
                # Size order: use position_size_pct of available cash
                invest_amount = portfolio.cash * self._position_size_pct
                quantity = invest_amount / (current_price * (1 + self._slippage_rate + self._commission_rate))

                if quantity > 0:
                    order = broker.execute_order(
                        symbol=symbol,
                        quantity=quantity,
                        market_price=current_price,
                        side=OrderSide.BUY,
                        timestamp=current_time,
                    )
                    if order and portfolio.apply_order(order):
                        trade = self._build_trade_record(sim_run_id, order)
                        executed_trades.append(trade)

            elif signal == Signal.SELL and position is not None:
                order = broker.execute_order(
                    symbol=symbol,
                    quantity=position.quantity,
                    market_price=current_price,
                    side=OrderSide.SELL,
                    timestamp=current_time,
                    avg_buy_price=position.avg_buy_price,
                )
                if order and portfolio.apply_order(order):
                    trade = self._build_trade_record(sim_run_id, order)
                    executed_trades.append(trade)

        # ── 4. Close any open position at final bar ───────────────────────────
        final_bar = df.iloc[-1]
        final_price = float(final_bar["close"])
        final_time = final_bar["timestamp"].to_pydatetime() if hasattr(
            final_bar["timestamp"], "to_pydatetime"
        ) else final_bar["timestamp"]

        position = portfolio.get_position(symbol)
        if position is not None:
            close_order = broker.execute_order(
                symbol=symbol,
                quantity=position.quantity,
                market_price=final_price,
                side=OrderSide.SELL,
                timestamp=final_time,
                avg_buy_price=position.avg_buy_price,
            )
            if close_order and portfolio.apply_order(close_order):
                executed_trades.append(self._build_trade_record(sim_run_id, close_order))

        # ── 5. Compute final metrics ─────────────────────────────────────────
        final_value = portfolio.cash  # All positions closed
        total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
        sharpe = self._compute_sharpe(equity_curve)
        max_drawdown = self._compute_max_drawdown(equity_curve)
        win_rate, total_trades = self._compute_win_rate(executed_trades)

        # ── 6. Persist results to DB ─────────────────────────────────────────
        if save_to_db and sim_run_id != -1:
            self._save_results(
                sim_run_id=sim_run_id,
                trades=executed_trades,
                portfolio_snapshots=portfolio.snapshots,
                final_value=final_value,
                total_return_pct=total_return_pct,
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
            )

        logger.info(
            "Simulation complete: return=%.2f%%, Sharpe=%.2f, drawdown=%.2f%%, trades=%d",
            total_return_pct,
            sharpe,
            max_drawdown,
            total_trades,
        )

        return SimResult(
            sim_run_id=sim_run_id,
            strategy_name=strategy.name,
            symbol=symbol,
            interval=interval,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            trades=[t.to_dict() for t in executed_trades],
            equity_curve=equity_curve,
        )

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _create_sim_run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        interval: str,
        initial_capital: float,
        start_date: str,
        end_date: str,
    ) -> Optional[SimRun]:
        """Create and persist a SimRun record in the database."""
        session = get_session()
        try:
            sim_run = SimRun(
                strategy=strategy.name,
                symbol=symbol,
                interval=interval,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
            )
            session.add(sim_run)
            session.commit()
            session.refresh(sim_run)
            logger.debug("Created SimRun id=%d", sim_run.id)
            return sim_run
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to create SimRun: %s", exc)
            return None
        finally:
            session.close()

    def _build_trade_record(self, sim_run_id: int, order) -> Trade:
        """Build a Trade ORM object from an ExecutedOrder."""
        return Trade(
            sim_run_id=sim_run_id,
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=order.executed_price,
            fees=order.fees,
            pnl=order.pnl,
        )

    def _save_results(
        self,
        sim_run_id: int,
        trades: list[Trade],
        portfolio_snapshots: list,
        final_value: float,
        total_return_pct: float,
        sharpe: float,
        max_drawdown: float,
        win_rate: float,
    ) -> None:
        """Persist all trades, snapshots, and final metrics to the database."""
        session = get_session()
        try:
            # Update SimRun with final metrics
            db_run = session.get(SimRun, sim_run_id)
            if db_run:
                db_run.final_value = final_value
                db_run.total_return_pct = total_return_pct
                db_run.sharpe_ratio = sharpe
                db_run.max_drawdown_pct = max_drawdown
                db_run.win_rate = win_rate
                db_run.total_trades = len(trades)

            # Bulk-insert trades
            session.bulk_save_objects(trades)

            # Persist portfolio snapshots (sample every 10 bars to limit DB size)
            sampled = portfolio_snapshots[::10] if len(portfolio_snapshots) > 100 else portfolio_snapshots
            snapshot_rows = [
                PortfolioModel(
                    sim_run_id=sim_run_id,
                    timestamp=snap.timestamp,
                    cash=snap.cash,
                    total_value=snap.total_value,
                    positions_json=str(
                        {sym: {"qty": pos.quantity, "avg_price": pos.avg_buy_price}
                         for sym, pos in snap.positions.items()}
                    ),
                )
                for snap in sampled
            ]
            session.bulk_save_objects(snapshot_rows)
            session.commit()

        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to save simulation results: %s", exc)
        finally:
            session.close()

    # ─── Metrics ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(equity_curve: list[tuple[datetime, float]]) -> float:
        """
        Compute annualized Sharpe ratio from the equity curve.
        Assumes risk-free rate of 0.

        Args:
            equity_curve: List of (timestamp, total_value) tuples.

        Returns:
            Annualized Sharpe ratio, or 0.0 if insufficient data.
        """
        if len(equity_curve) < 2:
            return 0.0

        values = np.array([v for _, v in equity_curve], dtype=float)
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize: assume hourly bars → 252 trading days × 7 hours/day
        annualization_factor = np.sqrt(252 * 7)
        sharpe = (np.mean(returns) / np.std(returns)) * annualization_factor
        return float(round(sharpe, 4))

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[tuple[datetime, float]]) -> float:
        """
        Compute maximum drawdown percentage from the equity curve.

        Args:
            equity_curve: List of (timestamp, total_value) tuples.

        Returns:
            Maximum drawdown as a positive percentage (e.g., 15.3 for -15.3%).
        """
        if len(equity_curve) < 2:
            return 0.0

        values = np.array([v for _, v in equity_curve], dtype=float)
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return float(round(max_drawdown, 4))

    @staticmethod
    def _compute_win_rate(trades: list[Trade]) -> tuple[float, int]:
        """
        Compute win rate from completed SELL trades.

        Args:
            trades: List of Trade ORM objects.

        Returns:
            Tuple of (win_rate, total_sell_trades).
        """
        sell_trades = [t for t in trades if t.side == "SELL"]
        if not sell_trades:
            return 0.0, 0

        winning = sum(1 for t in sell_trades if t.pnl > 0)
        win_rate = winning / len(sell_trades)
        return float(round(win_rate, 4)), len(sell_trades)
