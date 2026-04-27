"""
TradeFlow — SQLAlchemy Database Models
Defines ORM tables: SimRun, Trade, Portfolio, PriceCache, AISignal, BotDecisionLog
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    __allow_unmapped__ = True
    pass


class SimRun(Base):
    """
    Represents a single simulation run.

    Attributes:
        id: Primary key.
        strategy: Strategy name used for this run.
        symbol: Traded asset symbol (e.g., 'AAPL', 'MC.PA').
        interval: OHLCV bar interval (e.g., '1h', '15m').
        initial_capital: Starting capital in account currency.
        start_date: Simulation start date.
        end_date: Simulation end date.
        final_value: Final portfolio value.
        total_return_pct: Total return percentage.
        sharpe_ratio: Sharpe ratio of the run.
        max_drawdown_pct: Maximum drawdown percentage.
        win_rate: Ratio of winning trades (0.0–1.0).
        total_trades: Total number of trades executed.
        created_at: Timestamp when the run was created.
    """
    __tablename__ = "sim_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    strategy: str = Column(String(64), nullable=False)
    symbol: str = Column(String(16), nullable=False)
    interval: str = Column(String(8), nullable=False)
    initial_capital: float = Column(Float, nullable=False)
    start_date: str = Column(String(32), nullable=True)
    end_date: str = Column(String(32), nullable=True)

    # Results (populated after simulation completes)
    final_value: float = Column(Float, nullable=True)
    total_return_pct: float = Column(Float, nullable=True)
    sharpe_ratio: float = Column(Float, nullable=True)
    max_drawdown_pct: float = Column(Float, nullable=True)
    win_rate: float = Column(Float, nullable=True)
    total_trades: int = Column(Integer, nullable=True)

    # Live trading fields
    is_live: bool = Column(Boolean, default=False, nullable=False)
    status: str = Column(String(16), default="completed", nullable=False)  # running|completed|stopped
    last_tick_at: datetime = Column(DateTime, nullable=True)

    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    trades: list[Trade] = relationship("Trade", back_populates="sim_run", cascade="all, delete-orphan")
    portfolio_snapshots: list[Portfolio] = relationship(
        "Portfolio", back_populates="sim_run", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Serialize SimRun to dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "interval": self.interval,
            "initial_capital": self.initial_capital,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "final_value": self.final_value,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "is_live": self.is_live,
            "status": self.status,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Trade(Base):
    """
    Represents a single executed trade within a simulation run.

    Attributes:
        id: Primary key.
        sim_run_id: Foreign key referencing SimRun.
        timestamp: Trade execution timestamp.
        symbol: Asset symbol.
        side: 'BUY' or 'SELL'.
        quantity: Number of shares/units.
        price: Execution price (post-slippage).
        fees: Commission fees paid.
        pnl: Realized P&L for SELL trades (0.0 for BUY).
    """
    __tablename__ = "trades"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    sim_run_id: int = Column(Integer, ForeignKey("sim_runs.id"), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False)
    symbol: str = Column(String(16), nullable=False)
    side: str = Column(String(4), nullable=False)  # 'BUY' | 'SELL'
    quantity: float = Column(Float, nullable=False)
    price: float = Column(Float, nullable=False)
    fees: float = Column(Float, nullable=False, default=0.0)
    pnl: float = Column(Float, nullable=False, default=0.0)
    reason: str = Column(Text, nullable=True, default="")  # Why the bot made this trade

    sim_run: SimRun = relationship("SimRun", back_populates="trades")

    def to_dict(self) -> dict:
        """Serialize Trade to dictionary."""
        return {
            "id": self.id,
            "sim_run_id": self.sim_run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "fees": self.fees,
            "pnl": self.pnl,
            "reason": self.reason or "",
        }


class Portfolio(Base):
    """
    Represents a periodic portfolio snapshot for equity curve plotting.

    Attributes:
        id: Primary key.
        sim_run_id: Foreign key referencing SimRun.
        timestamp: Snapshot timestamp.
        cash: Available cash.
        total_value: Total portfolio value (cash + positions).
        positions_json: JSON-serialized open positions dict.
    """
    __tablename__ = "portfolio_snapshots"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    sim_run_id: int = Column(Integer, ForeignKey("sim_runs.id"), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False)
    cash: float = Column(Float, nullable=False)
    total_value: float = Column(Float, nullable=False)
    positions_json: str = Column(Text, nullable=False, default="{}")

    sim_run: SimRun = relationship("SimRun", back_populates="portfolio_snapshots")

    @property
    def positions(self) -> dict:
        """Deserialize positions from JSON string."""
        return json.loads(self.positions_json)

    @positions.setter
    def positions(self, value: dict) -> None:
        """Serialize positions to JSON string."""
        self.positions_json = json.dumps(value)

    def to_dict(self) -> dict:
        """Serialize Portfolio snapshot to dictionary."""
        return {
            "id": self.id,
            "sim_run_id": self.sim_run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cash": self.cash,
            "total_value": self.total_value,
            "positions": self.positions,
        }


class PriceCache(Base):
    """
    Caches downloaded OHLCV price bars to avoid redundant API calls.

    Attributes:
        id: Primary key.
        symbol: Asset symbol.
        interval: Bar interval.
        timestamp: Bar open timestamp.
        open: Opening price.
        high: Highest price.
        low: Lowest price.
        close: Closing price.
        volume: Traded volume.
    """
    __tablename__ = "price_cache"
    __table_args__ = (
        Index("ix_price_cache_sym_int_ts", "symbol", "interval", "timestamp"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(16), nullable=False)
    interval: str = Column(String(8), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False)
    open: float = Column(Float, nullable=False)
    high: float = Column(Float, nullable=False)
    low: float = Column(Float, nullable=False)
    close: float = Column(Float, nullable=False)
    volume: float = Column(Float, nullable=False)

    def to_dict(self) -> dict:
        """Serialize PriceCache entry to dictionary."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class AISignal(Base):
    """
    Persiste les signaux et decisions IA calcules par le scheduler.
    """
    __tablename__ = "ai_signals"
    __table_args__ = (
        Index("ix_ai_sym_int_ts", "symbol", "interval", "computed_at"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(16), nullable=False)
    interval: str = Column(String(8), nullable=False, default="1d")
    mode: str = Column(String(16), nullable=False, default="autonomous")
    score: float = Column(Float, nullable=True)
    action: str = Column(String(8), nullable=True)
    confidence: float = Column(Float, nullable=True)
    position_size_pct: float = Column(Float, nullable=True)
    stop_loss_pct: float = Column(Float, nullable=True)
    take_profit_pct: float = Column(Float, nullable=True)
    rationale: str = Column(Text, nullable=True)
    key_risks: str = Column(Text, nullable=True)
    sources_json: str = Column(Text, nullable=True)
    computed_at: datetime = Column(DateTime, nullable=False,
                                   default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "interval": self.interval,
            "mode": self.mode,
            "score": self.score,
            "action": self.action,
            "confidence": self.confidence,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "rationale": self.rationale,
            "key_risks": self.key_risks,
            "sources": json.loads(self.sources_json) if self.sources_json else [],
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
        }


class BotDecisionLog(Base):
    """
    Memoire persistante du bot — pourquoi il a (ou pas) agi sur chaque ticker.
    """
    __tablename__ = "bot_decisions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    sim_run_id: int = Column(Integer, ForeignKey("sim_runs.id"), nullable=False)
    symbol: str = Column(String(16), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False,
                                  default=lambda: datetime.now(timezone.utc))
    action: str = Column(String(8), nullable=False)
    reason: str = Column(Text, nullable=False)
    price: float = Column(Float, nullable=True)
    ai_action: str = Column(String(8), nullable=True)
    ai_confidence: float = Column(Float, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sim_run_id": self.sim_run_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "action": self.action,
            "reason": self.reason,
            "price": self.price,
            "ai_action": self.ai_action,
            "ai_confidence": self.ai_confidence,
        }
