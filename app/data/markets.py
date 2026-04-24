"""
TradeFlow — Market Hours & Status
Tracks opening/closing times for major exchanges.
Determines if markets are open, and when the bot should start/stop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ── Exchange definitions ────────────────────────────────────────────────────────

@dataclass
class Exchange:
    name: str
    tz: str                    # IANA timezone
    open_time: time            # Local market open
    close_time: time           # Local market close
    index_ticker: str          # yfinance ticker for the index
    index_name: str             # Display name

EXCHANGES = [
    Exchange("Euronext Paris",  "Europe/Paris",   time(9, 0),  time(17, 30), "^FCHI", "CAC 40"),
    Exchange("Xetra Frankfurt", "Europe/Berlin",  time(9, 0),  time(17, 30), "^GDAXI", "DAX 40"),
    Exchange("NASDAQ",          "US/Eastern",      time(9, 30), time(16, 0),  "^IXIC",  "NASDAQ"),
    Exchange("NYSE",            "US/Eastern",      time(9, 30), time(16, 0),  "^DJI",   "Dow Jones"),
]


def is_market_open(exchange: Exchange, now: Optional[datetime] = None) -> bool:
    """Check if a market is currently open."""
    if now is None:
        now = datetime.utcnow()
    tz = ZoneInfo(exchange.tz)
    local_now = now.astimezone(tz)
    # Saturday=5, Sunday=6
    if local_now.weekday() >= 5:
        return False
    market_open = local_now.replace(hour=exchange.open_time.hour, minute=exchange.open_time.minute, second=0, microsecond=0)
    market_close = local_now.replace(hour=exchange.close_time.hour, minute=exchange.close_time.minute, second=0, microsecond=0)
    return market_open <= local_now <= market_close


def any_market_open(now: Optional[datetime] = None) -> bool:
    """Check if at least one market is currently open."""
    return any(is_market_open(ex, now) for ex in EXCHANGES)


def next_market_event(now: Optional[datetime] = None) -> tuple[str, datetime]:
    """Return ('open'|'close', datetime) of the next market open or close event."""
    if now is None:
        now = datetime.utcnow()
    if any_market_open(now):
        # Find the soonest close
        soonest = None
        for ex in EXCHANGES:
            if is_market_open(ex, now):
                tz = ZoneInfo(ex.tz)
                local_now = now.astimezone(tz)
                close_local = local_now.replace(
                    hour=ex.close_time.hour, minute=ex.close_time.minute, second=0, microsecond=0
                )
                if close_local > local_now:
                    close_utc = close_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                    if soonest is None or close_utc < soonest:
                        soonest = close_utc
        return ("close", soonest) if soonest else ("close", now + timedelta(hours=1))
    else:
        # Find the soonest open
        soonest = None
        for ex in EXCHANGES:
            tz = ZoneInfo(ex.tz)
            local_now = now.astimezone(tz)
            # Try today
            open_local = local_now.replace(
                hour=ex.open_time.hour, minute=ex.open_time.minute, second=0, microsecond=0
            )
            if open_local > local_now and local_now.weekday() < 5:
                open_utc = open_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                if soonest is None or open_utc < soonest:
                    soonest = open_utc
            else:
                # Next weekday
                days_ahead = 1
                while (local_now.weekday() + days_ahead) % 7 >= 5:
                    days_ahead += 1
                next_day = local_now + timedelta(days=days_ahead)
                open_local = next_day.replace(
                    hour=ex.open_time.hour, minute=ex.open_time.minute, second=0, microsecond=0
                )
                open_utc = open_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                if soonest is None or open_utc < soonest:
                    soonest = open_utc
        return ("open", soonest) if soonest else ("open", now + timedelta(hours=8))


def get_market_status(exchange: Exchange) -> dict:
    """Get status info for one exchange including index data and price change."""
    open_flag = is_market_open(exchange)
    index_data = _fetch_index_price(exchange.index_ticker)
    result = {
        "name": exchange.name,
        "index_name": exchange.index_name,
        "index_ticker": exchange.index_ticker,
        "open": open_flag,
    }
    if index_data is not None:
        result["price"] = index_data["price"]
        result["prev_close"] = index_data["prev_close"]
    else:
        result["price"] = None
        result["prev_close"] = None
    return result


def _fetch_index_price(ticker: str) -> Optional[dict]:
    """Fetch the latest price and previous close for a market index.

    Returns dict with 'price' and 'prev_close', or None on failure.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist.iloc[-1]["Close"])
            prev_close = float(hist.iloc[-2]["Close"]) if len(hist) >= 2 else price
            return {"price": price, "prev_close": prev_close}
    except Exception:
        pass
    return None


def get_all_market_statuses() -> list[dict]:
    """Get status for all tracked exchanges."""
    return [get_market_status(ex) for ex in EXCHANGES]