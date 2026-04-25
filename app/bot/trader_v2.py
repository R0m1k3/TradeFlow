"""
TradeFlow v2 — orchestrated trader.

Decision flow per tick (per symbol):
  1. RegimeDetector    → regime + exposure_multiplier
       - if BEAR → no new longs, manage exits only
  2. Strategy primary  → BUY / HOLD / SELL signal
       - DualMomentum (monthly rebalance), or
       - PullbackTrend (intraday/daily)
  3. MetaLabeler       → P(success), filter weak signals
  4. RiskManager       → approve & size & set stop
  5. Execute order     → broker
  6. Manage stops      → trailing stop ratchet + circuit breakers

This module replaces live_trader.py for v2 strategies. The v1 LiveTrader
remains for backward compatibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.meta_label.meta_labeler import MetaLabeler, build_features
from app.regime.detector import Regime, RegimeDetector
from app.risk.manager import PortfolioSnapshot, RiskManager, RiskVerdict
from app.risk.stops import compute_atr, trailing_stop_price
from app.simulator.broker import OrderSide, VirtualBroker
from app.simulator.portfolio import Portfolio
from app.strategies_v2.dual_momentum import DualMomentumStrategy, DualSignal
from app.strategies_v2.pullback_trend import PullbackSignal, PullbackTrendStrategy

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Per-position tracking for active stops."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    initial_stop: float
    current_stop: float
    high_water: float = 0.0       # peak price since entry (for trailing)
    atr_at_entry: float = 0.0


@dataclass
class TraderConfig:
    benchmark_symbol: str = "SPY"
    use_meta_labeler: bool = True
    meta_labeler_path: Optional[str] = None
    primary_strategy: str = "pullback_trend"   # or "dual_momentum"
    universe: list[str] = field(default_factory=list)
    bars_period: str = "1y"
    bars_interval: str = "1d"
    initial_capital: float = 10_000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    atr_trail_multiplier: float = 2.5


class TraderV2:
    def __init__(self, config: TraderConfig) -> None:
        self.config = config
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager()
        self.broker = VirtualBroker(config.commission_rate, config.slippage_rate)
        self.portfolio = Portfolio(config.initial_capital)
        self.position_states: dict[str, PositionState] = {}

        # Strategies
        self.dual_momentum = DualMomentumStrategy()
        self.pullback = PullbackTrendStrategy()

        # Meta labeler (optional)
        self.meta_labeler: Optional[MetaLabeler] = None
        if config.use_meta_labeler and config.meta_labeler_path:
            try:
                self.meta_labeler = MetaLabeler.load(config.meta_labeler_path)
                logger.info("MetaLabeler loaded from %s", config.meta_labeler_path)
            except Exception as exc:
                logger.warning("Could not load MetaLabeler: %s — running without it", exc)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """One full decision cycle. Returns a status dict."""
        actions: list[dict] = []

        # 1. Regime check
        regime_signal = self._detect_regime()
        if regime_signal is None:
            return {"error": "regime detection failed", "actions": actions}

        # 2. Manage existing positions FIRST (stops, trailing, exits)
        for sym in list(self.position_states.keys()):
            action = self._manage_position(sym)
            if action:
                actions.append(action)

        # 3. New entries — only if regime allows
        if regime_signal.regime == Regime.BEAR:
            logger.info("Regime=BEAR — no new entries")
            return self._build_status(regime_signal, actions)

        for sym in self.config.universe:
            if sym in self.position_states:
                continue  # already in this name
            action = self._evaluate_entry(sym, regime_signal)
            if action:
                actions.append(action)

        return self._build_status(regime_signal, actions)

    # ── Regime ────────────────────────────────────────────────────────────────

    def _detect_regime(self):
        try:
            df = fetch_ohlcv(self.config.benchmark_symbol, interval="1d", period="2y")
            if df is None or df.empty:
                return None
            return self.regime_detector.detect(df["close"])
        except Exception as exc:
            logger.error("Regime detection failed: %s", exc)
            return None

    # ── Entry evaluation ──────────────────────────────────────────────────────

    def _evaluate_entry(self, symbol: str, regime_signal) -> Optional[dict]:
        df = fetch_ohlcv(symbol, interval=self.config.bars_interval, period=self.config.bars_period)
        if df is None or df.empty or len(df) < 250:
            return None
        df = add_all_indicators(df)

        # Primary signal
        if self.config.primary_strategy == "pullback_trend":
            decision = self.pullback.generate(df, in_position=False)
            if decision.signal != PullbackSignal.LONG:
                return None
            entry = decision.entry
            atr = decision.atr
            primary_reason = decision.reason
        else:
            # Dual momentum needs the full universe — handled elsewhere
            return None

        # Meta-labeler filter
        if self.meta_labeler is not None:
            features = build_features(df, len(df) - 1, regime_signal.to_dict())
            if features:
                act, p_success = self.meta_labeler.should_act(features)
                if not act:
                    logger.info("Meta-labeler rejected %s (P=%.2f)", symbol, p_success)
                    return {
                        "symbol": symbol,
                        "action": "skip_meta",
                        "p_success": p_success,
                    }
                logger.info("Meta-labeler approved %s (P=%.2f)", symbol, p_success)

        # Risk manager — sizing and circuit breakers
        snapshot = self._portfolio_snapshot()
        verdict = self.risk_manager.evaluate_entry(
            symbol=symbol,
            entry=entry,
            atr=atr,
            portfolio=snapshot,
        )
        if not verdict.approved:
            return {
                "symbol": symbol,
                "action": "rejected",
                "reason": verdict.reason,
                "verdict": verdict.verdict.value,
            }

        # Apply regime exposure multiplier
        size = verdict.size * regime_signal.exposure_multiplier
        if size < self.risk_manager.min_shares:
            return None

        # Execute
        order = self.broker.execute_order(
            symbol=symbol,
            quantity=size,
            market_price=entry,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )
        if not order or not self.portfolio.apply_order(order):
            return {"symbol": symbol, "action": "execution_failed"}

        self.position_states[symbol] = PositionState(
            symbol=symbol,
            quantity=size,
            entry_price=order.executed_price,
            entry_time=order.timestamp,
            initial_stop=verdict.initial_stop,
            current_stop=verdict.initial_stop,
            high_water=order.executed_price,
            atr_at_entry=atr,
        )
        logger.info(
            "✅ BUY %s qty=%.4f @ %.2f stop=%.2f reason=%s",
            symbol, size, order.executed_price, verdict.initial_stop, primary_reason,
        )
        return {
            "symbol": symbol,
            "action": "buy",
            "quantity": size,
            "price": order.executed_price,
            "stop": verdict.initial_stop,
            "reason": primary_reason,
        }

    # ── Position management ───────────────────────────────────────────────────

    def _manage_position(self, symbol: str) -> Optional[dict]:
        ps = self.position_states.get(symbol)
        if ps is None:
            return None

        df = fetch_ohlcv(symbol, interval=self.config.bars_interval, period=self.config.bars_period)
        if df is None or df.empty:
            return None
        df = add_all_indicators(df)

        current_price = float(df["close"].iloc[-1])
        atr = float(compute_atr(df, 14).iloc[-1])
        if atr <= 0:
            atr = ps.atr_at_entry

        # Update high-water
        if current_price > ps.high_water:
            ps.high_water = current_price

        # Trailing stop ratchet
        ps.current_stop = trailing_stop_price(
            initial_stop=ps.current_stop,
            current_high=ps.high_water,
            atr=atr,
            multiplier=self.config.atr_trail_multiplier,
            side="long",
        )

        # Stop hit?
        if self.risk_manager.should_exit_on_stop(current_price, ps.current_stop):
            return self._close_position(symbol, current_price, reason="stop_hit")

        # Optional strategy-driven exit (pullback reverted)
        if self.config.primary_strategy == "pullback_trend":
            decision = self.pullback.generate(df, in_position=True)
            if decision.signal == PullbackSignal.EXIT_LONG:
                return self._close_position(symbol, current_price, reason="strategy_exit")

        return None

    def _close_position(self, symbol: str, price: float, reason: str) -> dict:
        ps = self.position_states.pop(symbol, None)
        if ps is None:
            return {"symbol": symbol, "action": "no_position"}

        order = self.broker.execute_order(
            symbol=symbol,
            quantity=ps.quantity,
            market_price=price,
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc),
            avg_buy_price=ps.entry_price,
        )
        if order:
            self.portfolio.apply_order(order)
            self.risk_manager.on_position_exit(symbol, order.pnl)
            logger.info(
                "🔻 SELL %s qty=%.4f @ %.2f P&L=%.2f reason=%s",
                symbol, ps.quantity, price, order.pnl, reason,
            )
            return {
                "symbol": symbol,
                "action": "sell",
                "quantity": ps.quantity,
                "price": price,
                "pnl": order.pnl,
                "reason": reason,
            }
        return {"symbol": symbol, "action": "close_failed"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _portfolio_snapshot(self) -> PortfolioSnapshot:
        positions = {}
        for sym, ps in self.position_states.items():
            positions[sym] = {
                "qty": ps.quantity,
                "avg_price": ps.entry_price,
                "current_price": ps.entry_price,  # could be updated with live price
            }
        equity = self.portfolio.cash + sum(
            p["qty"] * p["current_price"] for p in positions.values()
        )
        return PortfolioSnapshot(
            cash=self.portfolio.cash,
            equity=equity,
            positions=positions,
        )

    def _build_status(self, regime_signal, actions) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime_signal.to_dict(),
            "risk_status": self.risk_manager.status(),
            "positions": {
                sym: {
                    "qty": ps.quantity,
                    "entry": ps.entry_price,
                    "stop": ps.current_stop,
                    "high_water": ps.high_water,
                }
                for sym, ps in self.position_states.items()
            },
            "actions": actions,
            "cash": self.portfolio.cash,
        }
