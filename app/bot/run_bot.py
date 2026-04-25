#!/usr/bin/env python3
"""
TradeFlow — Live Bot Entry Point
Runs the live trading loop. Auto-starts when first market opens,
pauses when last market closes.

Tick interval is read from the BOT_TICK_SECONDS env var (default: 3600 = 1h).
Set BOT_TICK_SECONDS=60 for testing (1-minute ticks).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.bot.live_trader import LiveTrader
from app.bot.trader_v2 import TraderConfig, TraderV2
from app.data.markets import any_market_open, next_market_event
from app.database.session import init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradeflow.bot")


def _build_trader():
    """Build either v1 LiveTrader or v2 TraderV2 based on env."""
    use_v2 = os.environ.get("BOT_VERSION", "v2").lower() == "v2"
    if not use_v2:
        logger.info("Using v1 LiveTrader (legacy)")
        return LiveTrader(
            commission_rate=float(os.environ.get("BOT_COMMISSION", "0.001")),
            slippage_rate=float(os.environ.get("BOT_SLIPPAGE", "0.0005")),
            position_size_pct=float(os.environ.get("BOT_POSITION_SIZE", "0.95")),
            data_period=os.environ.get("BOT_DATA_PERIOD", "3mo"),
        )

    logger.info("Using v2 TraderV2 (regime + risk + meta-label)")
    universe_str = os.environ.get(
        "BOT_UNIVERSE",
        "AAPL,MSFT,GOOG,AMZN,META,NVDA,SPY",
    )
    universe = [s.strip().upper() for s in universe_str.split(",") if s.strip()]

    meta_path = os.environ.get("BOT_META_LABELER_PATH", "data/meta_labeler.pkl")
    if not Path(meta_path).exists():
        logger.warning("Meta-labeler not found at %s — running without it", meta_path)
        meta_path = None

    config = TraderConfig(
        benchmark_symbol=os.environ.get("BOT_BENCHMARK", "SPY"),
        use_meta_labeler=meta_path is not None,
        meta_labeler_path=meta_path,
        primary_strategy=os.environ.get("BOT_STRATEGY", "pullback_trend"),
        universe=universe,
        bars_period=os.environ.get("BOT_DATA_PERIOD", "1y"),
        bars_interval=os.environ.get("BOT_INTERVAL", "1d"),
        initial_capital=float(os.environ.get("BOT_INITIAL_CAPITAL", "10000")),
        commission_rate=float(os.environ.get("BOT_COMMISSION", "0.001")),
        slippage_rate=float(os.environ.get("BOT_SLIPPAGE", "0.0005")),
    )
    return TraderV2(config)


def main() -> None:
    tick_interval = int(os.environ.get("BOT_TICK_SECONDS", "3600"))

    logger.info("=" * 60)
    logger.info("TradeFlow Live Bot starting")
    logger.info("   Tick interval : %ds", tick_interval)
    logger.info("=" * 60)

    init_database()

    trader = _build_trader()

    running = True

    def _handle_stop(sig, frame):
        nonlocal running
        logger.info("Signal %s received — stopping.", sig)
        running = False

    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except AttributeError:
        pass  # SIGTERM not available on Windows
    try:
        signal.signal(signal.SIGINT, _handle_stop)
    except AttributeError:
        pass

    logger.info("Bot ready. Auto-starts when markets open, pauses when they close.")

    while running:
        now = datetime.now(timezone.utc)

        if any_market_open(now):
            # Markets are open — execute ticks
            try:
                result = trader.tick()
                if isinstance(result, dict) and result.get("actions"):
                    for action in result["actions"]:
                        logger.info("Action: %s", action)
                if isinstance(result, dict) and "regime" in result:
                    r = result["regime"]
                    logger.info(
                        "Regime=%s | exposure=%.1fx | vol=%.1f%%",
                        r["regime"], r["exposure_multiplier"],
                        r["realized_vol_annual"] * 100,
                    )
                logger.info("Tick complete — markets open")
            except Exception as exc:
                logger.error("Unhandled tick error: %s", exc, exc_info=True)

            # Sleep until next tick
            deadline = time.monotonic() + tick_interval
            while running and time.monotonic() < deadline:
                time.sleep(1)
        else:
            # Markets are closed — sleep until next market opens
            event_type, event_time = next_market_event(now)
            wait_seconds = max(1, (event_time - now).total_seconds())
            wait_minutes = int(wait_seconds // 60)
            logger.info(
                "Marches fermes. Prochaine ouverture dans ~%d min (a %s UTC)",
                wait_minutes,
                event_time.strftime("%H:%M"),
            )

            # Sleep in chunks, checking every 30s
            deadline = time.monotonic() + min(wait_seconds, 3600)
            while running and time.monotonic() < deadline:
                time.sleep(30)

    logger.info("TradeFlow Bot stopped.")


if __name__ == "__main__":
    main()