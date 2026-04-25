"""RiskManager — the gatekeeper. Every order flows through here.

Use:
    rm = RiskManager()
    verdict = rm.evaluate_entry(symbol, capital, entry, atr, portfolio_state)
    if verdict.approved:
        qty = verdict.size
        stop = verdict.initial_stop
        execute_order(symbol, qty, stop)
    else:
        logger.info("Entry rejected: %s", verdict.reason)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from app.risk.circuit_breakers import CircuitBreaker, CircuitBreakerState, BreakerLevel
from app.risk.kill_switch import KillSwitch
from app.risk.sizing import position_size_by_risk
from app.risk.stops import atr_stop

logger = logging.getLogger(__name__)


class RiskVerdict(str, Enum):
    APPROVED = "approved"
    REJECTED_KILL_SWITCH = "rejected_kill_switch"
    REJECTED_CIRCUIT_BREAKER = "rejected_circuit_breaker"
    REJECTED_MAX_POSITIONS = "rejected_max_positions"
    REJECTED_SECTOR_CAP = "rejected_sector_cap"
    REJECTED_SYMBOL_COOLDOWN = "rejected_symbol_cooldown"
    REJECTED_INSUFFICIENT_CASH = "rejected_insufficient_cash"
    REJECTED_SIZE_TOO_SMALL = "rejected_size_too_small"
    REJECTED_BAD_STOP = "rejected_bad_stop"


@dataclass
class RiskDecision:
    verdict: RiskVerdict
    size: float = 0.0
    initial_stop: float = 0.0
    reason: str = ""
    breaker_state: CircuitBreakerState | None = None

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED


@dataclass
class PortfolioSnapshot:
    """What the RiskManager needs to know about the current portfolio."""
    cash: float
    equity: float                                       # cash + value of positions
    positions: dict[str, dict] = field(default_factory=dict)  # {sym: {qty, avg_price, sector}}

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    def sector_exposure(self, equity: float) -> dict[str, float]:
        """Dollar value per sector as fraction of equity."""
        exposures: dict[str, float] = {}
        for p in self.positions.values():
            sector = p.get("sector", "Unknown")
            dollar_value = p["qty"] * p.get("current_price", p["avg_price"])
            exposures[sector] = exposures.get(sector, 0.0) + dollar_value
        if equity <= 0:
            return {k: 0.0 for k in exposures}
        return {k: v / equity for k, v in exposures.items()}


class CooldownTracker:
    """Prevents rapid re-entry on the same symbol (avoids whipsaws)."""
    def __init__(self, cooldown_minutes: int = 30) -> None:
        self._cd = cooldown_minutes
        self._last_exit: dict[str, datetime] = {}

    def on_exit(self, symbol: str) -> None:
        self._last_exit[symbol] = datetime.now(timezone.utc)

    def in_cooldown(self, symbol: str) -> bool:
        last = self._last_exit.get(symbol)
        if last is None:
            return False
        age = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return age < self._cd


class RiskManager:
    """Central gatekeeper for all trade decisions."""

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.15,
        max_sector_pct: float = 0.30,
        max_positions: int = 5,
        atr_stop_multiplier: float = 2.5,
        cooldown_minutes: int = 30,
        min_shares: float = 0.001,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_positions = max_positions
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_shares = min_shares

        self.circuit_breaker = CircuitBreaker()
        self.kill_switch = KillSwitch()
        self.cooldown = CooldownTracker(cooldown_minutes)

    # ── Pre-trade check ───────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        symbol: str,
        entry: float,
        atr: float,
        portfolio: PortfolioSnapshot,
        sector: str | None = None,
    ) -> RiskDecision:
        """
        Decide whether to approve a new LONG entry and at what size.
        Returns a RiskDecision — only execute if .approved is True.
        """
        # 1. Kill switch (external emergency)
        if self.kill_switch.is_active:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_KILL_SWITCH,
                reason=f"kill-switch active: {self.kill_switch.reason}",
            )

        # 2. Circuit breakers (internal equity-based halts)
        cb_state = self.circuit_breaker.update(portfolio.equity)
        if cb_state.is_halted:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_CIRCUIT_BREAKER,
                reason=cb_state.reason,
                breaker_state=cb_state,
            )

        # 3. Max concurrent positions
        if portfolio.num_positions >= self.max_positions and symbol not in portfolio.positions:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_MAX_POSITIONS,
                reason=f"max positions reached ({self.max_positions})",
            )

        # 4. Symbol cooldown (no re-entry right after exit)
        if self.cooldown.in_cooldown(symbol):
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_SYMBOL_COOLDOWN,
                reason=f"cooldown active on {symbol}",
            )

        # 5. Compute stop & size
        if atr <= 0:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_BAD_STOP,
                reason="ATR is zero or negative",
            )
        stop = atr_stop(entry, atr, self.atr_stop_multiplier, side="long")
        if stop >= entry or stop <= 0:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_BAD_STOP,
                reason=f"invalid stop {stop:.2f} for entry {entry:.2f}",
            )

        size = position_size_by_risk(
            capital=portfolio.equity,
            entry=entry,
            stop=stop,
            risk_pct=self.risk_per_trade,
            max_position_pct=self.max_position_pct,
        )

        if size < self.min_shares:
            return RiskDecision(
                verdict=RiskVerdict.REJECTED_SIZE_TOO_SMALL,
                reason=f"computed size {size:.6f} < min {self.min_shares}",
            )

        cost = size * entry
        if cost > portfolio.cash:
            # Rescale to available cash
            size = portfolio.cash / entry * 0.99  # 1% slack for fees/slippage
            if size < self.min_shares:
                return RiskDecision(
                    verdict=RiskVerdict.REJECTED_INSUFFICIENT_CASH,
                    reason=f"cash {portfolio.cash:.2f} < cost {cost:.2f}",
                )

        # 6. Sector concentration check
        if sector:
            exposures = portfolio.sector_exposure(portfolio.equity)
            current_sector = exposures.get(sector, 0.0)
            projected = current_sector + (size * entry) / max(portfolio.equity, 1e-9)
            if projected > self.max_sector_pct:
                return RiskDecision(
                    verdict=RiskVerdict.REJECTED_SECTOR_CAP,
                    reason=f"sector {sector} would exceed {self.max_sector_pct:.0%} (projected {projected:.1%})",
                )

        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            size=size,
            initial_stop=stop,
            breaker_state=cb_state,
        )

    # ── Post-trade accounting ─────────────────────────────────────────────────

    def on_position_exit(self, symbol: str, pnl: float) -> None:
        """Notify the manager that a position was closed."""
        self.cooldown.on_exit(symbol)
        logger.info("Exit %s | P&L=%.2f | cooldown started", symbol, pnl)

    # ── Stop management ───────────────────────────────────────────────────────

    def should_exit_on_stop(self, current_price: float, current_stop: float, side: str = "long") -> bool:
        """Is the current price beyond the stop?"""
        if side == "long":
            return current_price <= current_stop
        return current_price >= current_stop

    # ── Introspection ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "kill_switch_active": self.kill_switch.is_active,
            "kill_switch_reason": self.kill_switch.reason,
            "circuit_breaker": {
                "tripped_level": self.circuit_breaker.state.tripped_level.value,
                "halted": self.circuit_breaker.state.is_halted,
                "reason": self.circuit_breaker.state.reason,
                "peak_equity": self.circuit_breaker.peak_equity,
            },
            "limits": {
                "risk_per_trade": self.risk_per_trade,
                "max_position_pct": self.max_position_pct,
                "max_sector_pct": self.max_sector_pct,
                "max_positions": self.max_positions,
                "atr_stop_multiplier": self.atr_stop_multiplier,
            },
        }
