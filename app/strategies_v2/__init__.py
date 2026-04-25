"""TradeFlow v2 strategies — evidence-based, non-overlapping signals.

Each strategy is orthogonal in its edge source:
  - DualMomentum     : trend (Antonacci) — profits from persistence
  - CrossSectional   : relative strength — profits from dispersion
  - PullbackTrend    : tactical mean-reversion inside confirmed uptrend

Strategies output a primary Signal. The meta-labeler (Layer 4) decides
whether to act on it. The RiskManager (Layer 5) sizes and protects.
"""
from app.strategies_v2.dual_momentum import DualMomentumStrategy
from app.strategies_v2.cross_sectional import CrossSectionalMomentumStrategy
from app.strategies_v2.pullback_trend import PullbackTrendStrategy

__all__ = [
    "DualMomentumStrategy",
    "CrossSectionalMomentumStrategy",
    "PullbackTrendStrategy",
]
