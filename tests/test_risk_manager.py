"""Tests for RiskManager — the critical safety layer."""
from __future__ import annotations

import pandas as pd
import pytest

from app.risk.circuit_breakers import BreakerLevel, CircuitBreaker
from app.risk.kill_switch import KillSwitch
from app.risk.manager import (
    PortfolioSnapshot,
    RiskManager,
    RiskVerdict,
)
from app.risk.sizing import kelly_fractional, position_size_by_risk
from app.risk.stops import atr_stop, compute_atr, trailing_stop_price


# ── Stops ────────────────────────────────────────────────────────────────────


def test_compute_atr_basic():
    df = pd.DataFrame({
        "high":  [10, 11, 12, 13, 14, 15, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "low":   [9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "close": [10, 11, 12, 13, 14, 15, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    })
    atr = compute_atr(df, period=14)
    assert len(atr) == len(df)
    assert atr.iloc[-1] > 0
    assert atr.iloc[-1] < 10  # sanity: can't exceed max TR


def test_atr_stop_long():
    assert atr_stop(100, atr=2.0, multiplier=2.5, side="long") == 95.0


def test_atr_stop_short():
    assert atr_stop(100, atr=2.0, multiplier=2.5, side="short") == 105.0


def test_trailing_stop_ratchets_up_only():
    # Start: stop at 95, entry was 100, ATR=2
    initial = 95.0
    # Price goes to 105 → stop should move up to 105 - 2.5*2 = 100
    new_stop = trailing_stop_price(initial, current_high=105, atr=2.0, multiplier=2.5, side="long")
    assert new_stop == 100.0
    # Price drops back to 98 → stop should NOT move down
    same_stop = trailing_stop_price(new_stop, current_high=98, atr=2.0, multiplier=2.5, side="long")
    assert same_stop == 100.0  # stayed at 100


# ── Sizing ───────────────────────────────────────────────────────────────────


def test_position_size_by_risk_basic():
    # 10_000 capital, 1% risk = 100$ at risk
    # Entry 100, stop 95 → risk/share = 5 → 100/5 = 20 shares
    shares = position_size_by_risk(10_000, entry=100, stop=95, risk_pct=0.01)
    assert shares == pytest.approx(20.0)


def test_position_size_capped_by_max_position_pct():
    # Same setup but tight stop (risk/share=1) would suggest 100 shares = 10_000$ = 100% of capital
    # Cap at 15% → 1_500/100 = 15 shares
    shares = position_size_by_risk(10_000, entry=100, stop=99, risk_pct=0.01, max_position_pct=0.15)
    assert shares == pytest.approx(15.0)


def test_position_size_zero_on_invalid_input():
    assert position_size_by_risk(0, 100, 95) == 0.0
    assert position_size_by_risk(10_000, 100, 100) == 0.0  # stop == entry
    assert position_size_by_risk(10_000, 0, 95) == 0.0


def test_kelly_fractional_positive_edge():
    # 60% win rate, 2:1 win/loss → Full Kelly = 0.6 - 0.4/2 = 0.4
    # Quarter Kelly = 0.10
    assert kelly_fractional(0.6, 2.0, fraction=0.25) == pytest.approx(0.10)


def test_kelly_fractional_no_edge():
    # 40% win rate, 1:1 → negative Kelly, returns 0 (don't trade)
    assert kelly_fractional(0.4, 1.0) == 0.0


def test_kelly_capped_at_25pct():
    # Absurd edge: 95% win rate, 10:1 → Full Kelly ~ 0.95 - 0.05/10 = 0.945
    # Quarter would be 0.236 but capped at 0.25
    assert kelly_fractional(0.95, 10, fraction=0.5) == 0.25


# ── Circuit Breakers ─────────────────────────────────────────────────────────


def test_circuit_breaker_starts_clean():
    cb = CircuitBreaker()
    state = cb.update(10_000)
    assert state.tripped_level == BreakerLevel.NONE
    assert not state.is_halted


def test_circuit_breaker_trips_on_daily_loss():
    cb = CircuitBreaker(daily_loss_threshold=0.03)
    cb.update(10_000)  # initial
    state = cb.update(9_600)  # -4% → trips DAILY
    assert state.tripped_level == BreakerLevel.DAILY
    assert state.is_halted


def test_circuit_breaker_trips_on_max_dd():
    cb = CircuitBreaker(max_drawdown_threshold=0.15)
    cb.update(10_000)
    cb.update(11_000)  # new peak
    state = cb.update(9_000)  # -18% from peak → trips MAX_DD
    assert state.tripped_level == BreakerLevel.MAX_DD
    assert state.is_halted
    # MAX_DD requires manual reset
    cb.update(11_500)  # even back to new high, still halted
    assert cb.state.is_halted


def test_circuit_breaker_manual_reset():
    cb = CircuitBreaker(max_drawdown_threshold=0.15)
    cb.update(10_000)
    cb.update(8_000)  # -20% → trips
    assert cb.state.is_halted
    cb.manual_reset()
    assert not cb.state.is_halted


# ── Kill Switch ──────────────────────────────────────────────────────────────


def test_kill_switch_inactive_by_default(tmp_path):
    ks = KillSwitch(path=tmp_path / "KILL")
    assert not ks.is_active


def test_kill_switch_activate_and_deactivate(tmp_path):
    ks = KillSwitch(path=tmp_path / "KILL")
    ks.activate("test emergency")
    assert ks.is_active
    assert ks.reason == "test emergency"
    ks.deactivate()
    assert not ks.is_active


# ── RiskManager integration ──────────────────────────────────────────────────


def make_portfolio(cash=10_000, positions=None):
    positions = positions or {}
    equity = cash + sum(
        p["qty"] * p.get("current_price", p["avg_price"])
        for p in positions.values()
    )
    return PortfolioSnapshot(cash=cash, equity=equity, positions=positions)


def test_risk_manager_approves_normal_entry(tmp_path):
    rm = RiskManager()
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    portfolio = make_portfolio(cash=10_000)
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2.0, portfolio=portfolio)
    assert verdict.approved
    assert verdict.size > 0
    assert verdict.initial_stop == 95.0  # 100 - 2.5 * 2


def test_risk_manager_rejects_when_kill_switch_active(tmp_path):
    rm = RiskManager()
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    rm.kill_switch.activate("tests")
    portfolio = make_portfolio()
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2, portfolio=portfolio)
    assert verdict.verdict == RiskVerdict.REJECTED_KILL_SWITCH


def test_risk_manager_rejects_on_too_many_positions(tmp_path):
    rm = RiskManager(max_positions=2)
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    portfolio = make_portfolio(
        cash=10_000,
        positions={
            "MSFT": {"qty": 10, "avg_price": 300, "current_price": 310},
            "GOOG": {"qty": 5, "avg_price": 150, "current_price": 155},
        },
    )
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2, portfolio=portfolio)
    assert verdict.verdict == RiskVerdict.REJECTED_MAX_POSITIONS


def test_risk_manager_rejects_on_sector_cap(tmp_path):
    rm = RiskManager(max_sector_pct=0.30)
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    # Already 25% in Tech, trying to add another 15% → 40% > 30% cap
    portfolio = make_portfolio(
        cash=10_000,
        positions={"MSFT": {"qty": 10, "avg_price": 300, "current_price": 300, "sector": "Tech"}},
    )
    # Equity = 10_000 + 3_000 = 13_000; MSFT = 3_000/13_000 = 23%
    # Adding ~15% = 2_000$ of AAPL → 38% tech exposure
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=0.5, portfolio=portfolio, sector="Tech")
    assert verdict.verdict == RiskVerdict.REJECTED_SECTOR_CAP


def test_risk_manager_rejects_on_circuit_breaker(tmp_path):
    rm = RiskManager()
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    # Simulate equity drop that trips daily breaker
    rm.circuit_breaker.update(10_000)
    rm.circuit_breaker.update(9_600)  # -4% → DAILY trips
    portfolio = make_portfolio(cash=9_600)
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2, portfolio=portfolio)
    assert verdict.verdict == RiskVerdict.REJECTED_CIRCUIT_BREAKER


def test_risk_manager_symbol_cooldown(tmp_path):
    rm = RiskManager(cooldown_minutes=30)
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    rm.on_position_exit("AAPL", pnl=-50)
    portfolio = make_portfolio()
    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2, portfolio=portfolio)
    assert verdict.verdict == RiskVerdict.REJECTED_SYMBOL_COOLDOWN


def test_risk_manager_1pct_rule_on_normal_trade(tmp_path):
    """Property test: if entry is approved and stop is hit, loss ≤ 1% of equity."""
    rm = RiskManager(risk_per_trade=0.01)
    rm.kill_switch = KillSwitch(path=tmp_path / "K")
    portfolio = make_portfolio(cash=10_000)

    verdict = rm.evaluate_entry("AAPL", entry=100, atr=2.0, portfolio=portfolio)
    assert verdict.approved

    # If stop hit: loss = size * (entry - stop)
    loss_at_stop = verdict.size * (100 - verdict.initial_stop)
    loss_pct = loss_at_stop / portfolio.equity
    # Should be close to 1% (may be slightly under due to position cap)
    assert loss_pct <= 0.0101  # tiny tolerance
