"""
TradeFlow — Unit Tests: Simulator (Broker, Portfolio, Engine)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.simulator.broker import ExecutedOrder, OrderSide, VirtualBroker
from app.simulator.portfolio import Portfolio
from app.simulator.engine import SimulationEngine
from app.strategies.base import Signal
from app.strategies.sma_crossover import SmaCrossoverStrategy


# ─── VirtualBroker Tests ──────────────────────────────────────────────────────

class TestVirtualBroker:
    @pytest.fixture
    def broker(self) -> VirtualBroker:
        return VirtualBroker(commission_rate=0.001, slippage_rate=0.0005)

    def test_buy_order_applies_slippage(self, broker: VirtualBroker) -> None:
        """BUY orders should execute slightly above market price."""
        order = broker.execute_order(
            symbol="AAPL",
            quantity=10,
            market_price=100.0,
            side=OrderSide.BUY,
            timestamp=datetime.utcnow(),
        )
        assert order is not None
        assert order.executed_price > 100.0
        assert order.executed_price == pytest.approx(100.0 * 1.0005, rel=1e-6)

    def test_sell_order_applies_slippage(self, broker: VirtualBroker) -> None:
        """SELL orders should execute slightly below market price."""
        order = broker.execute_order(
            symbol="AAPL",
            quantity=10,
            market_price=100.0,
            side=OrderSide.SELL,
            timestamp=datetime.utcnow(),
            avg_buy_price=90.0,
        )
        assert order is not None
        assert order.executed_price < 100.0
        assert order.executed_price == pytest.approx(100.0 * 0.9995, rel=1e-6)

    def test_fees_calculated_correctly(self, broker: VirtualBroker) -> None:
        """Fees should be 0.1% of executed gross value."""
        order = broker.execute_order(
            symbol="AAPL",
            quantity=10,
            market_price=100.0,
            side=OrderSide.BUY,
            timestamp=datetime.utcnow(),
        )
        expected_gross = order.executed_price * 10
        expected_fees = expected_gross * 0.001
        assert order.fees == pytest.approx(expected_fees, rel=1e-6)

    def test_zero_quantity_returns_none(self, broker: VirtualBroker) -> None:
        order = broker.execute_order(
            symbol="AAPL",
            quantity=0,
            market_price=100.0,
            side=OrderSide.BUY,
            timestamp=datetime.utcnow(),
        )
        assert order is None

    def test_negative_price_returns_none(self, broker: VirtualBroker) -> None:
        order = broker.execute_order(
            symbol="AAPL",
            quantity=10,
            market_price=-5.0,
            side=OrderSide.BUY,
            timestamp=datetime.utcnow(),
        )
        assert order is None

    def test_invalid_commission_raises(self) -> None:
        with pytest.raises(ValueError):
            VirtualBroker(commission_rate=0.5)

    def test_sell_pnl_computed_when_avg_price_provided(self, broker: VirtualBroker) -> None:
        """P&L for SELL should be positive when selling above average buy price."""
        order = broker.execute_order(
            symbol="AAPL",
            quantity=10,
            market_price=110.0,
            side=OrderSide.SELL,
            timestamp=datetime.utcnow(),
            avg_buy_price=100.0,
        )
        assert order is not None
        assert order.pnl > 0


# ─── Portfolio Tests ──────────────────────────────────────────────────────────

class TestPortfolio:
    @pytest.fixture
    def portfolio(self) -> Portfolio:
        return Portfolio(initial_capital=10_000.0)

    @pytest.fixture
    def buy_order(self) -> ExecutedOrder:
        return ExecutedOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10.0,
            requested_price=100.0,
            executed_price=100.05,
            fees=1.0,
            timestamp=datetime(2024, 1, 1, 9, 0),
        )

    @pytest.fixture
    def sell_order(self) -> ExecutedOrder:
        return ExecutedOrder(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10.0,
            requested_price=110.0,
            executed_price=109.945,
            fees=1.0995,
            timestamp=datetime(2024, 1, 2, 9, 0),
            pnl=90.05,
        )

    def test_initial_cash(self, portfolio: Portfolio) -> None:
        assert portfolio.cash == 10_000.0

    def test_buy_reduces_cash(self, portfolio: Portfolio, buy_order: ExecutedOrder) -> None:
        initial_cash = portfolio.cash
        portfolio.apply_order(buy_order)
        assert portfolio.cash < initial_cash

    def test_buy_creates_position(self, portfolio: Portfolio, buy_order: ExecutedOrder) -> None:
        portfolio.apply_order(buy_order)
        pos = portfolio.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == pytest.approx(10.0)

    def test_sell_after_buy_removes_position(
        self,
        portfolio: Portfolio,
        buy_order: ExecutedOrder,
        sell_order: ExecutedOrder,
    ) -> None:
        portfolio.apply_order(buy_order)
        portfolio.apply_order(sell_order)
        assert portfolio.get_position("AAPL") is None

    def test_sell_without_position_rejected(
        self, portfolio: Portfolio, sell_order: ExecutedOrder
    ) -> None:
        result = portfolio.apply_order(sell_order)
        assert result is False

    def test_buy_exceeding_cash_rejected(self, portfolio: Portfolio) -> None:
        expensive_order = ExecutedOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000.0,
            requested_price=100.0,
            executed_price=100.05,
            fees=10_000.0,
            timestamp=datetime.utcnow(),
        )
        result = portfolio.apply_order(expensive_order)
        assert result is False
        assert portfolio.cash == 10_000.0  # Unchanged

    def test_invalid_initial_capital_raises(self) -> None:
        with pytest.raises(ValueError):
            Portfolio(initial_capital=-1000)

    def test_snapshot_recorded(self, portfolio: Portfolio, buy_order: ExecutedOrder) -> None:
        ts = datetime(2024, 1, 1, 10, 0)
        portfolio.apply_order(buy_order)
        portfolio.take_snapshot(ts, {"AAPL": 105.0})
        assert len(portfolio.snapshots) == 1
        snap = portfolio.snapshots[0]
        assert snap.total_value > 0

    def test_total_value_includes_positions(
        self, portfolio: Portfolio, buy_order: ExecutedOrder
    ) -> None:
        portfolio.apply_order(buy_order)
        total = portfolio.get_total_value({"AAPL": 120.0})
        # Cash + (10 shares × $120)
        expected = portfolio.cash + 10 * 120.0
        assert total == pytest.approx(expected, rel=1e-4)


# ─── SimulationEngine Integration Tests ───────────────────────────────────────

class TestSimulationEngine:
    """
    Integration tests using mocked yfinance data to avoid external API calls.
    """

    @pytest.fixture
    def mock_ohlcv_df(self) -> pd.DataFrame:
        """Synthetic 200-bar OHLCV DataFrame."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = close.clip(1)

        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": [1_000_000.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

    def test_run_returns_sim_result(self, mock_ohlcv_df: pd.DataFrame, tmp_path) -> None:
        """Full simulation run with mocked data should return a valid SimResult."""
        engine = SimulationEngine()
        strategy = SmaCrossoverStrategy(fast_period=10, slow_period=30)

        with (
            patch("app.simulator.engine.fetch_ohlcv", return_value=mock_ohlcv_df),
            patch("app.simulator.engine.init_database"),
            patch("app.simulator.engine.get_session") as mock_session_factory,
        ):
            # Mock session
            mock_session = MagicMock()
            mock_session_factory.return_value = mock_session
            mock_sim_run = MagicMock()
            mock_sim_run.id = 1
            mock_session.get.return_value = mock_sim_run
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            mock_session.refresh.side_effect = lambda obj: setattr(obj, "id", 1)

            result = engine.run(
                strategy=strategy,
                symbol="AAPL",
                interval="1h",
                period="3mo",
                initial_capital=10_000.0,
            )

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.initial_capital == 10_000.0
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert 0.0 <= result.win_rate <= 1.0

    def test_run_fails_gracefully_on_no_data(self) -> None:
        """Engine should return None when data fetch fails."""
        engine = SimulationEngine()
        strategy = SmaCrossoverStrategy()

        with patch("app.simulator.engine.fetch_ohlcv", return_value=None), \
             patch("app.simulator.engine.init_database"):
            result = engine.run(
                strategy=strategy,
                symbol="INVALID_TICKER_XYZ",
                interval="1h",
                period="3mo",
                initial_capital=10_000.0,
            )

        assert result is None

    def test_sharpe_computation(self) -> None:
        """Sharpe ratio should be 0 for flat equity curve."""
        flat_curve = [(datetime(2024, 1, i + 1), 10_000.0) for i in range(50)]
        sharpe = SimulationEngine._compute_sharpe(flat_curve)
        assert sharpe == 0.0

    def test_max_drawdown_no_loss(self) -> None:
        """Monotonically increasing equity should have 0% drawdown."""
        growing_curve = [(datetime(2024, 1, 1), 10_000 + i * 100) for i in range(50)]
        dd = SimulationEngine._compute_max_drawdown(growing_curve)
        assert dd == 0.0

    def test_win_rate_computation(self) -> None:
        """Win rate should correctly identify profitable SELL trades."""
        from app.database.models import Trade

        trades = []
        for pnl in [100, -50, 200, -30, 150]:
            t = Trade()
            t.side = "SELL"
            t.pnl = pnl
            trades.append(t)

        win_rate, total = SimulationEngine._compute_win_rate(trades)
        assert win_rate == pytest.approx(3 / 5, rel=1e-6)
        assert total == 5
