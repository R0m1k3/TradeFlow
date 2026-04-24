"""
TradeFlow — Portfolio Manager
Manages virtual positions, cash balance, and P&L tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.simulator.broker import ExecutedOrder, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents an open position for a single asset.

    Attributes:
        symbol: Asset ticker.
        quantity: Number of units held.
        avg_buy_price: Volume-weighted average buy price.
        total_cost: Total cash spent (excluding fees, included in pnl).
    """
    symbol: str
    quantity: float
    avg_buy_price: float
    total_cost: float


@dataclass
class PortfolioSnapshot:
    """
    Immutable snapshot of portfolio state at a specific point in time.

    Attributes:
        timestamp: Snapshot time.
        cash: Available cash.
        positions_value: Mark-to-market value of all open positions.
        total_value: cash + positions_value.
        positions: Dict of symbol → Position.
    """
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    positions: dict[str, Position]


class Portfolio:
    """
    Virtual portfolio managing cash, positions, and P&L.

    All operations assume a single currency (USD or EUR depending on asset).

    Args:
        initial_capital: Starting cash balance (default: 10,000).
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self._initial_capital: float = initial_capital
        self._cash: float = initial_capital
        self._positions: dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._snapshots: list[PortfolioSnapshot] = []

    # ─── Public API ────────────────────────────────────────────────────────────

    def apply_order(self, order: ExecutedOrder) -> bool:
        """
        Apply an executed broker order to update cash and positions.

        Args:
            order: The ExecutedOrder returned by VirtualBroker.execute_order().

        Returns:
            True if the order was applied, False if rejected (e.g., insufficient cash).
        """
        if order.side == OrderSide.BUY:
            return self._process_buy(order)
        return self._process_sell(order)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Return the current open position for a symbol, or None.

        Args:
            symbol: Asset ticker symbol.

        Returns:
            Position object or None if no position is held.
        """
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Return all currently open positions."""
        return dict(self._positions)

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """
        Calculate current total portfolio value (cash + mark-to-market positions).

        Args:
            current_prices: Dict mapping symbol → current market price.

        Returns:
            Total portfolio value in account currency.
        """
        positions_value = sum(
            pos.quantity * current_prices.get(symbol, pos.avg_buy_price)
            for symbol, pos in self._positions.items()
        )
        return self._cash + positions_value

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """
        Calculate unrealized P&L from open positions.

        Args:
            current_prices: Dict mapping symbol → current market price.

        Returns:
            Sum of unrealized P&L across all open positions.
        """
        return sum(
            (current_prices.get(symbol, pos.avg_buy_price) - pos.avg_buy_price) * pos.quantity
            for symbol, pos in self._positions.items()
        )

    def take_snapshot(self, timestamp: datetime, current_prices: dict[str, float]) -> None:
        """
        Record a portfolio snapshot for equity curve plotting.

        Args:
            timestamp: Bar timestamp.
            current_prices: Dict mapping symbol → current close price.
        """
        positions_value = sum(
            pos.quantity * current_prices.get(symbol, pos.avg_buy_price)
            for symbol, pos in self._positions.items()
        )
        total_value = self._cash + positions_value

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions_value=positions_value,
            total_value=total_value,
            positions=dict(self._positions),
        )
        self._snapshots.append(snapshot)

    # ─── Properties ────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current available cash balance."""
        return self._cash

    @property
    def initial_capital(self) -> float:
        """Starting capital."""
        return self._initial_capital

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return self._realized_pnl

    @property
    def snapshots(self) -> list[PortfolioSnapshot]:
        """List of all portfolio snapshots (chronological)."""
        return list(self._snapshots)

    # ─── Private helpers ────────────────────────────────────────────────────────

    def _process_buy(self, order: ExecutedOrder) -> bool:
        """Process a BUY order: deduct cash, add/update position."""
        total_cost = order.total_cost  # executed_price * qty + fees

        if total_cost > self._cash:
            logger.warning(
                "BUY rejected: insufficient cash (%.2f < %.2f required)",
                self._cash,
                total_cost,
            )
            return False

        self._cash -= total_cost

        symbol = order.symbol
        if symbol in self._positions:
            existing = self._positions[symbol]
            new_qty = existing.quantity + order.quantity
            # Volume-weighted average price
            new_avg = (
                (existing.avg_buy_price * existing.quantity)
                + (order.executed_price * order.quantity)
            ) / new_qty
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_buy_price=new_avg,
                total_cost=existing.total_cost + total_cost,
            )
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=order.quantity,
                avg_buy_price=order.executed_price,
                total_cost=total_cost,
            )

        logger.debug("BUY applied: cash=%.2f, position=%s", self._cash, self._positions[symbol])
        return True

    def _process_sell(self, order: ExecutedOrder) -> bool:
        """Process a SELL order: receive cash, reduce/close position."""
        symbol = order.symbol
        position = self._positions.get(symbol)

        if position is None:
            logger.warning("SELL rejected: no position held for %s", symbol)
            return False

        if order.quantity > position.quantity:
            logger.warning(
                "SELL rejected: quantity %.4f exceeds held %.4f for %s",
                order.quantity,
                position.quantity,
                symbol,
            )
            return False

        # Cash inflow = executed_price * qty - fees
        cash_received = order.executed_price * order.quantity - order.fees
        self._cash += cash_received
        self._realized_pnl += order.pnl

        new_qty = position.quantity - order.quantity
        if new_qty < 1e-9:  # Floating point: treat as fully closed
            del self._positions[symbol]
            logger.debug("Position closed for %s, cash=%.2f", symbol, self._cash)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_buy_price=position.avg_buy_price,
                total_cost=position.total_cost * (new_qty / position.quantity),
            )

        return True
