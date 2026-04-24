#!/usr/bin/env python3
"""
TradeFlow — Live Bot Entry Point
Runs the live trading loop continuously. Designed to run as a Docker service.

The bot checks for an active live session in the DB each tick.
If one is found, it executes trades. If not, it waits.

Tick interval is read from the BOT_TICK_SECONDS env var (default: 3600 = 1h).
Set BOT_TICK_SECONDS=60 for testing (1-minute ticks).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.bot.live_trader import LiveTrader
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
    logger.info("🤖  TradeFlow Live Bot starting")
    logger.info("   Tick interval : %ds (%s)", tick_interval, _fmt_interval(tick_interval))
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
        logger.info("Signal %s received — stopping after current tick.", sig)
        running = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    logger.info("Bot ready. Waiting for an active live session (start one from the WebUI).")

    while running:
        try:
            trader.tick()
        except Exception as exc:
            logger.error("Unhandled tick error: %s", exc, exc_info=True)

        if not running:
            break

        logger.info("Sleeping %ds until next tick…", tick_interval)
        # Sleep in small chunks so we react to SIGTERM quickly
        deadline = time.monotonic() + tick_interval
        while running and time.monotonic() < deadline:
            time.sleep(1)

    logger.info("🛑  TradeFlow Bot stopped.")


def _fmt_interval(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}min"
    return f"{seconds // 3600}h"


if __name__ == "__main__":
    main()
