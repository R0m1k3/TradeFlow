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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.bot.live_trader import LiveTrader
from app.data.markets import any_market_open, next_market_event
from app.database.session import init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradeflow.bot")


def main() -> None:
    tick_interval = int(os.environ.get("BOT_TICK_SECONDS", "3600"))

    logger.info("=" * 60)
    logger.info("TradeFlow Live Bot starting")
    logger.info("   Tick interval : %ds", tick_interval)
    logger.info("=" * 60)

    init_database()

    trader = LiveTrader(
        commission_rate=float(os.environ.get("BOT_COMMISSION", "0.001")),
        slippage_rate=float(os.environ.get("BOT_SLIPPAGE", "0.0005")),
        position_size_pct=float(os.environ.get("BOT_POSITION_SIZE", "0.95")),
        data_period=os.environ.get("BOT_DATA_PERIOD", "3mo"),
    )

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
        now = datetime.utcnow()

        if any_market_open(now):
            # Markets are open — execute ticks
            try:
                trader.tick()
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