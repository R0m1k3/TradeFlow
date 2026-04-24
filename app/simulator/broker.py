"""
TradeFlow — Virtual Broker
Simulates order execution with realistic commission and slippage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Direction of a trade order."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type of order execution (Phase 1: MARKET only)."""
    MARKET = "MARKET"


@dataclass
class ExecutedOrder:
    """
    Result of a successfully executed order by the virtual broker.

    Attributes:
        symbol: Asset ticker symbol.
        side: BUY or SELL.
        quantity: Number of units traded.
        requested_price: Raw market price at execution bar.
        executed_price: Final price after slippage.
        fees: Commission fees charged.
        timestamp: Execution timestamp.
        pnl: Realized P&L (only meaningful for SELL orders).
    """
    symbol: str
    side: OrderSide
    quantity: float
    requested_price: float
    executed_price: float
    fees: float
    timestamp: datetime
    pnl: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cash outflow (BUY) or inflow (SELL) including fees."""
        gross = self.executed_price * self.quantity
        if self.side == OrderSide.BUY:
            return gross + self.fees
        return gross - self.fees


class VirtualBroker:
    """
    Virtual broker that simulates realistic order execution.

    Applies configurable commission and slippage to each order.
    Phase 1 supports MARKET orders only.

    Args:
        commission_rate: Fractional commission per trade (default: 0.001 = 0.1%).
        slippage_rate: Fractional slippage applied to execution price (default: 0.0005 = 0.05%).
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        if commission_rate < 0 or commission_rate > 0.1:
            raise ValueError(f"commission_rate must be in [0, 0.1], got {commission_rate}")
        if slippage_rate < 0 or slippage_rate > 0.05:
            raise ValueError(f"slippage_rate must be in [0, 0.05], got {slippage_rate}")

        self._commission_rate = commission_rate
        self._slippage_rate = slippage_rate

    def execute_order(
        self,
        symbol: str,
        quantity: float,
        market_price: float,
        side: OrderSide,
        timestamp: datetime,
        avg_buy_price: Optional[float] = None,
    ) -> Optional[ExecutedOrder]:
        """
        Execute a simulated MARKET order with slippage and commission.

        Args:
            symbol: Asset symbol.
            quantity: Number of units to trade (must be positive).
            market_price: Current close price at execution bar.
            side: OrderSide.BUY or OrderSide.SELL.
            timestamp: Current bar timestamp.
            avg_buy_price: Average buy price for P&L calculation on SELL (optional).

        Returns:
            ExecutedOrder on success, None if validation fails.
        """
        if quantity <= 0:
            logger.warning("Invalid quantity %.4f for %s — order rejected", quantity, symbol)
            return None

        if market_price <= 0:
            logger.warning("Invalid price %.4f for %s — order rejected", market_price, symbol)
            return None

        # Apply slippage: unfavorable price movement on execution
        if side == OrderSide.BUY:
            executed_price = market_price * (1 + self._slippage_rate)
        else:
            executed_price = market_price * (1 - self._slippage_rate)

        gross_value = executed_price * quantity
        fees = gross_value * self._commission_rate

        # Compute realized P&L for SELL orders
        pnl = 0.0
        if side == OrderSide.SELL and avg_buy_price is not None:
            buy_cost = avg_buy_price * quantity
            sell_proceeds = executed_price * quantity - fees
            pnl = sell_proceeds - buy_cost

        order = ExecutedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            requested_price=market_price,
            executed_price=executed_price,
            fees=fees,
            timestamp=timestamp,
            pnl=pnl,
        )

        logger.info(
            "%s %s x%.2f @ %.4f (slippage → %.4f, fees=%.4f)",
            side.value,
            symbol,
            quantity,
            market_price,
            executed_price,
            fees,
        )

        return order

    @property
    def commission_rate(self) -> float:
        """Commission rate as a fraction."""
        return self._commission_rate

    @property
    def slippage_rate(self) -> float:
        """Slippage rate as a fraction."""
        return self._slippage_rate
