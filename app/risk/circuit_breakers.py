"""Circuit breakers — the automatic safeties that halt trading on adverse moves.

Tripping logic:
  - Daily loss > 3%   → halt 24 h
  - Weekly loss > 7%  → halt until next Monday
  - Max DD > 15%      → stop bot, require human reset
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class BreakerLevel(str, Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MAX_DD = "max_dd"


@dataclass
class CircuitBreakerState:
    tripped_level: BreakerLevel = BreakerLevel.NONE
    tripped_at: datetime | None = None
    halt_until: datetime | None = None
    reason: str = ""

    @property
    def is_halted(self) -> bool:
        if self.tripped_level == BreakerLevel.NONE:
            return False
        if self.tripped_level == BreakerLevel.MAX_DD:
            return True  # requires manual reset
        if self.halt_until is None:
            return False
        return datetime.now(timezone.utc) < self.halt_until


@dataclass
class CircuitBreaker:
    """Monitors portfolio equity and trips when thresholds are breached."""
    daily_loss_threshold: float = 0.03      # 3 %
    weekly_loss_threshold: float = 0.07     # 7 %
    max_drawdown_threshold: float = 0.15    # 15 %

    # State
    state: CircuitBreakerState = field(default_factory=CircuitBreakerState)
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    weekly_start_equity: float = 0.0
    daily_reset_at: datetime | None = None
    weekly_reset_at: datetime | None = None

    def update(self, equity: float, now: datetime | None = None) -> CircuitBreakerState:
        """
        Called every tick with current portfolio equity.
        Updates peaks, resets daily/weekly anchors, and checks thresholds.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Initial seed
        if self.peak_equity == 0:
            self.peak_equity = equity
            self.daily_start_equity = equity
            self.weekly_start_equity = equity
            self.daily_reset_at = now
            self.weekly_reset_at = now

        # Update peak (for max DD)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Daily reset (every UTC midnight)
        if self.daily_reset_at is None or now.date() > self.daily_reset_at.date():
            self.daily_start_equity = equity
            self.daily_reset_at = now

        # Weekly reset (every Monday UTC)
        if self.weekly_reset_at is None or (
            now.isocalendar().week != self.weekly_reset_at.isocalendar().week
        ):
            self.weekly_start_equity = equity
            self.weekly_reset_at = now

        # Already halted? Check if halt expired
        if self.state.is_halted:
            return self.state
        if self.state.tripped_level != BreakerLevel.NONE and not self.state.is_halted:
            # Halt expired naturally, reset soft breakers (not MAX_DD)
            if self.state.tripped_level != BreakerLevel.MAX_DD:
                self.state = CircuitBreakerState()

        # Check MAX_DD first (hardest trip)
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd >= self.max_drawdown_threshold:
                self.state = CircuitBreakerState(
                    tripped_level=BreakerLevel.MAX_DD,
                    tripped_at=now,
                    halt_until=None,  # manual reset required
                    reason=f"Max DD {dd:.1%} ≥ {self.max_drawdown_threshold:.1%}",
                )
                return self.state

        # Check WEEKLY
        if self.weekly_start_equity > 0:
            weekly_loss = (self.weekly_start_equity - equity) / self.weekly_start_equity
            if weekly_loss >= self.weekly_loss_threshold:
                # Halt until next Monday
                days_to_monday = (7 - now.weekday()) % 7 or 7
                halt_until = (now + timedelta(days=days_to_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                self.state = CircuitBreakerState(
                    tripped_level=BreakerLevel.WEEKLY,
                    tripped_at=now,
                    halt_until=halt_until,
                    reason=f"Weekly loss {weekly_loss:.1%} ≥ {self.weekly_loss_threshold:.1%}",
                )
                return self.state

        # Check DAILY
        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity
            if daily_loss >= self.daily_loss_threshold:
                halt_until = now + timedelta(hours=24)
                self.state = CircuitBreakerState(
                    tripped_level=BreakerLevel.DAILY,
                    tripped_at=now,
                    halt_until=halt_until,
                    reason=f"Daily loss {daily_loss:.1%} ≥ {self.daily_loss_threshold:.1%}",
                )
                return self.state

        return self.state

    def manual_reset(self) -> None:
        """Manually reset after MAX_DD trip (requires human intervention)."""
        self.state = CircuitBreakerState()
