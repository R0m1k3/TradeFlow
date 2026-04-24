"""
TradeFlow — Live Trader
Executes real-time (simulated) trades on a continuous loop.
Each tick: fetch latest prices → run strategy → buy/sell if signal → persist to DB.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.database.models import Portfolio as PortfolioModel, SimRun, Trade
from app.database.session import get_session, init_database
from app.simulator.broker import ExecutedOrder, OrderSide, VirtualBroker
from app.simulator.portfolio import Portfolio, Position
from app.strategies.base import Signal
from app.strategies.composite_strategy import CompositeStrategy
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.sma_crossover import SmaCrossoverStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "composite": CompositeStrategy,
    "Composite [>0.7/<0.3]": CompositeStrategy,
    "sma_crossover": SmaCrossoverStrategy,
    "SMA Crossover (20/50)": SmaCrossoverStrategy,
    "rsi": RsiStrategy,
    "RSI (14) [30/70]": RsiStrategy,
    "macd": MacdStrategy,
    "MACD (12/26/9)": MacdStrategy,
}


class LiveTrader:
    """
    Runs a continuous paper-trading loop for one or more symbols.

    The bot always looks for an active live SimRun in the DB (status="running",
    is_live=True). If none exists, it waits. If one is found, it executes trades
    and updates portfolio snapshots each tick.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size_pct: float = 0.95,
        data_period: str = "3mo",
    ) -> None:
        self._commission_rate = commission_rate
        self._slippage_rate = slippage_rate
        self._position_size_pct = position_size_pct
        self._data_period = data_period
        self._broker = VirtualBroker(commission_rate, slippage_rate)

        # In-memory portfolio per active session (keyed by run_id)
        self._portfolio: Optional[Portfolio] = None
        self._active_run_id: Optional[int] = None

        init_database()

    # ─── Public API ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """
        One trading tick: find active live session, fetch data, run strategy,
        execute orders, persist state.
        """
        run = self._get_active_run()
        if run is None:
            logger.debug("No active live session found — waiting.")
            return

        # If session changed (restart or new session), rebuild portfolio
        if run.id != self._active_run_id:
            logger.info("Loading live session #%d (%s on %s)", run.id, run.strategy, run.symbol)
            self._portfolio = self._rebuild_portfolio(run)
            self._active_run_id = run.id

        symbols = [s.strip() for s in run.symbol.split(",")]
        strategy_cls = STRATEGY_MAP.get(run.strategy)
        if strategy_cls is None:
            logger.error("Unknown strategy: %s", run.strategy)
            return

        strategy = strategy_cls()
        current_prices: dict[str, float] = {}

        for symbol in symbols:
            try:
                self._process_symbol(symbol, strategy, run.interval, run.id)
            except Exception as exc:
                logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

            # Collect current price for snapshot
            df = fetch_ohlcv(symbol, interval=run.interval, period=self._data_period)
            if df is not None and not df.empty:
                current_prices[symbol] = float(df.iloc[-1]["close"])

        # Save portfolio snapshot
        if current_prices and self._portfolio:
            self._save_snapshot(run.id, current_prices)

        # Update last_tick_at
        self._update_tick_time(run.id)
        logger.info(
            "Tick complete — session #%d | cash=%.2f | positions=%s",
            run.id,
            self._portfolio.cash if self._portfolio else 0,
            list(self._portfolio.get_all_positions().keys()) if self._portfolio else [],
        )

    def stop_active_session(self) -> None:
        """Mark the active live session as stopped."""
        session = get_session()
        try:
            run = session.query(SimRun).filter_by(is_live=True, status="running").first()
            if run:
                run.status = "stopped"
                session.commit()
                logger.info("Live session #%d stopped.", run.id)
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to stop session: %s", exc)
        finally:
            session.close()
        self._portfolio = None
        self._active_run_id = None

    # ─── Core tick logic ───────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str, strategy, interval: str, run_id: int) -> None:
        """Fetch latest data, run strategy, execute order if signal."""
        df = fetch_ohlcv(symbol, interval=interval, period=self._data_period)
        if df is None or df.empty:
            logger.warning("No data for %s — skipping", symbol)
            return

        df = add_all_indicators(df)
        df.attrs["symbol"] = symbol  # Needed by CompositeStrategy for sentiment
        last_idx = len(df) - 1

        signal, reason = strategy.generate_signal(df, last_idx)
        current_price = float(df.iloc[-1]["close"])
        current_time = datetime.utcnow()

        logger.info(
            "%s → %s (%.2f) | %s",
            symbol,
            signal.value,
            current_price,
            reason or "no reason",
        )

        if signal == Signal.HOLD:
            return

        position = self._portfolio.get_position(symbol) if self._portfolio else None

        if signal == Signal.BUY and position is None:
            invest_amount = self._portfolio.cash * self._position_size_pct
            qty = invest_amount / (
                current_price * (1 + self._slippage_rate + self._commission_rate)
            )
            if qty < 0.001:
                logger.warning("BUY skipped: insufficient cash for %s", symbol)
                return

            order = self._broker.execute_order(
                symbol=symbol,
                quantity=qty,
                market_price=current_price,
                side=OrderSide.BUY,
                timestamp=current_time,
            )
            if order and self._portfolio.apply_order(order):
                self._save_trade(run_id, order, reason)
                logger.info("✅ BOUGHT %s: %.4f @ %.2f | reason: %s", symbol, qty, current_price, reason)

        elif signal == Signal.SELL and position is not None:
            order = self._broker.execute_order(
                symbol=symbol,
                quantity=position.quantity,
                market_price=current_price,
                side=OrderSide.SELL,
                timestamp=current_time,
                avg_buy_price=position.avg_buy_price,
            )
            if order and self._portfolio.apply_order(order):
                self._save_trade(run_id, order, reason)
                logger.info(
                    "✅ SOLD %s: %.4f @ %.2f | P&L: %.2f | reason: %s",
                    symbol, position.quantity, current_price, order.pnl, reason,
                )

    # ─── DB helpers ────────────────────────────────────────────────────────────

    def _get_active_run(self) -> Optional[SimRun]:
        """Return the currently active live SimRun, or None."""
        session = get_session()
        try:
            return session.query(SimRun).filter_by(is_live=True, status="running").first()
        finally:
            session.close()

    def _rebuild_portfolio(self, run: SimRun) -> Portfolio:
        """Reconstruct portfolio state by replaying all past trades of a live session."""
        portfolio = Portfolio(run.initial_capital)
        session = get_session()
        try:
            trades = (
                session.query(Trade)
                .filter_by(sim_run_id=run.id)
                .order_by(Trade.timestamp)
                .all()
            )
            for t in trades:
                order = ExecutedOrder(
                    symbol=t.symbol,
                    side=OrderSide(t.side),
                    quantity=t.quantity,
                    requested_price=t.price,
                    executed_price=t.price,
                    fees=t.fees,
                    timestamp=t.timestamp,
                    pnl=t.pnl,
                )
                portfolio.apply_order(order)
            logger.info(
                "Portfolio rebuilt from %d trades — cash=%.2f, positions=%s",
                len(trades),
                portfolio.cash,
                list(portfolio.get_all_positions().keys()),
            )
        finally:
            session.close()
        return portfolio

    def _save_trade(self, run_id: int, order: ExecutedOrder, reason: str) -> None:
        session = get_session()
        try:
            trade = Trade(
                sim_run_id=run_id,
                timestamp=order.timestamp,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.executed_price,
                fees=order.fees,
                pnl=order.pnl,
                reason=reason,
            )
            session.add(trade)
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to save trade: %s", exc)
        finally:
            session.close()

    def _save_snapshot(self, run_id: int, current_prices: dict[str, float]) -> None:
        if not self._portfolio:
            return
        total_value = self._portfolio.get_total_value(current_prices)
        positions_data = {
            sym: {"qty": pos.quantity, "avg_price": pos.avg_buy_price, "current_price": current_prices.get(sym, pos.avg_buy_price)}
            for sym, pos in self._portfolio.get_all_positions().items()
        }
        session = get_session()
        try:
            snap = PortfolioModel(
                sim_run_id=run_id,
                timestamp=datetime.utcnow(),
                cash=self._portfolio.cash,
                total_value=total_value,
                positions_json=json.dumps(positions_data),
            )
            session.add(snap)
            # Also update SimRun final_value for live display
            run = session.get(SimRun, run_id)
            if run:
                run.final_value = total_value
                run.total_return_pct = (
                    (total_value - run.initial_capital) / run.initial_capital * 100
                )
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to save snapshot: %s", exc)
        finally:
            session.close()

    def _update_tick_time(self, run_id: int) -> None:
        session = get_session()
        try:
            run = session.get(SimRun, run_id)
            if run:
                run.last_tick_at = datetime.utcnow()
                session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
        finally:
            session.close()


# ─── Session factory helpers (used by WebUI) ──────────────────────────────────

def create_live_session(
    strategy: str,
    symbols: list[str],
    interval: str,
    initial_capital: float,
) -> int:
    """
    Create a new live SimRun in the DB and mark it as running.
    Stops any previously running session first.

    Returns the new run_id.
    """
    session = get_session()
    try:
        # Stop any existing running live session
        existing = session.query(SimRun).filter_by(is_live=True, status="running").first()
        if existing:
            existing.status = "stopped"
            session.flush()

        run = SimRun(
            strategy=strategy,
            symbol=",".join(symbols),
            interval=interval,
            initial_capital=initial_capital,
            is_live=True,
            status="running",
            start_date=datetime.utcnow().date().isoformat(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        logger.info("Created live session #%d — %s on %s", run.id, strategy, symbols)
        return run.id
    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Failed to create live session: %s", exc)
        raise
    finally:
        session.close()


def stop_live_session() -> None:
    """Mark the active live session as stopped."""
    session = get_session()
    try:
        run = session.query(SimRun).filter_by(is_live=True, status="running").first()
        if run:
            run.status = "stopped"
            session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
    finally:
        session.close()


def get_active_live_session() -> Optional[dict]:
    """Return the active live SimRun as a dict, or None."""
    session = get_session()
    try:
        run = session.query(SimRun).filter_by(is_live=True, status="running").first()
        return run.to_dict() if run else None
    finally:
        session.close()
