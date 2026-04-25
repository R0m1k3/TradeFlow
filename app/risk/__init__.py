"""TradeFlow — Risk Management Module.

The non-negotiable safeguards layer. Every order must flow through here.
"""
from app.risk.manager import RiskManager, RiskDecision, RiskVerdict
from app.risk.stops import atr_stop, trailing_stop_price, compute_atr
from app.risk.sizing import position_size_by_risk, kelly_fractional
from app.risk.circuit_breakers import CircuitBreakerState, CircuitBreaker
from app.risk.kill_switch import KillSwitch

__all__ = [
    "RiskManager",
    "RiskDecision",
    "RiskVerdict",
    "atr_stop",
    "trailing_stop_price",
    "compute_atr",
    "position_size_by_risk",
    "kelly_fractional",
    "CircuitBreakerState",
    "CircuitBreaker",
    "KillSwitch",
]
