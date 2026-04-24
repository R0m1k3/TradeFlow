"""
TradeFlow — CLI Entry Point
Allows running simulations from the command line without WebUI.
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import yaml

from app.database.session import init_database
from app.simulator.engine import SimulationEngine
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.sma_crossover import SmaCrossoverStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradeflow.main")

STRATEGY_MAP: dict[str, type] = {
    "sma": SmaCrossoverStrategy,
    "rsi": RsiStrategy,
    "macd": MacdStrategy,
}


def load_config() -> dict:
    """Load global configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the simulation runner."""
    config = load_config()
    broker_cfg = config.get("broker", {})

    parser = argparse.ArgumentParser(
        description="TradeFlow — Algorithmic Trading Simulator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", type=str, default="AAPL", help="Asset ticker symbol")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGY_MAP.keys()),
        default="sma",
        help="Strategy to run",
    )
    parser.add_argument("--interval", type=str, default="1h", help="Bar interval (1h, 1d, 15m…)")
    parser.add_argument("--period", type=str, default="3mo", help="Historical period (1mo, 3mo, 1y…)")
    parser.add_argument(
        "--capital",
        type=float,
        default=config.get("default_capital", 10_000),
        help="Initial capital",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=broker_cfg.get("commission", 0.001),
        help="Commission rate (e.g., 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=broker_cfg.get("slippage", 0.0005),
        help="Slippage rate (e.g., 0.0005 = 0.05%%)",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point for running a simulation."""
    args = parse_args()

    init_database()
    logger.info("TradeFlow Simulation CLI")
    logger.info("Symbol: %s | Strategy: %s | Interval: %s | Period: %s | Capital: $%.2f",
                args.symbol, args.strategy, args.interval, args.period, args.capital)

    strategy_cls = STRATEGY_MAP[args.strategy]
    strategy = strategy_cls()

    engine = SimulationEngine(
        commission_rate=args.commission,
        slippage_rate=args.slippage,
    )

    result = engine.run(
        strategy=strategy,
        symbol=args.symbol,
        interval=args.interval,
        period=args.period,
        initial_capital=args.capital,
    )

    if result is None:
        logger.error("Simulation failed. Check symbol and interval.")
        sys.exit(1)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TRADEFLOW SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Strategy     : {result.strategy_name}")
    print(f"  Symbol       : {result.symbol}")
    print(f"  Interval     : {result.interval}")
    print(f"  Initial Cap  : ${result.initial_capital:>12,.2f}")
    print(f"  Final Value  : ${result.final_value:>12,.2f}")
    print(f"  Total Return : {result.total_return_pct:>+12.2f}%")
    print(f"  Sharpe Ratio : {result.sharpe_ratio:>12.4f}")
    print(f"  Max Drawdown : {result.max_drawdown_pct:>11.2f}%")
    print(f"  Win Rate     : {result.win_rate * 100:>11.1f}%")
    print(f"  Total Trades : {result.total_trades:>12d}")
    print(f"  DB Run ID    : {result.sim_run_id:>12d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
