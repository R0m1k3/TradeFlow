"""
TradeFlow — Finnhub Real-Time Client
WebSocket connection to Finnhub streaming trades.
Maintains a shared in-memory price store consumed by the SSE endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Latest prices: {symbol: {"price": float, "volume": float, "ts": int}}
_prices: dict[str, dict] = {}
_subscribers: list[Callable[[str, dict], None]] = []
_lock = threading.Lock()
_ws_thread: threading.Thread | None = None
_subscribed_symbols: set[str] = set()
_ws_app = None


def get_api_key() -> str | None:
    return os.environ.get("FINNHUB_API_KEY")


def get_price(symbol: str) -> dict | None:
    with _lock:
        return _prices.get(symbol)


def get_all_prices() -> dict[str, dict]:
    with _lock:
        return dict(_prices)


def add_subscriber(fn: Callable[[str, dict], None]) -> None:
    with _lock:
        _subscribers.append(fn)


def remove_subscriber(fn: Callable[[str, dict], None]) -> None:
    with _lock:
        _subscribers[:] = [s for s in _subscribers if s is not fn]


def _notify(symbol: str, data: dict) -> None:
    with _lock:
        subs = list(_subscribers)
    for fn in subs:
        try:
            fn(symbol, data)
        except Exception:
            pass


def subscribe_symbols(symbols: list[str]) -> None:
    global _ws_app, _ws_thread
    api_key = get_api_key()
    if not api_key:
        logger.warning("FINNHUB_API_KEY not set — real-time stream disabled")
        return

    new = set(symbols) - _subscribed_symbols
    if not new:
        return

    _subscribed_symbols.update(new)

    if _ws_app is not None:
        for sym in new:
            try:
                _ws_app.send(json.dumps({"type": "subscribe", "symbol": sym}))
            except Exception:
                pass
        return

    # Start WebSocket in background thread
    _ws_thread = threading.Thread(target=_run_ws, args=(api_key, set(symbols)), daemon=True)
    _ws_thread.start()


def _run_ws(api_key: str, symbols: set[str]) -> None:
    global _ws_app
    import websocket

    def on_open(ws):
        global _ws_app
        _ws_app = ws
        logger.info("Finnhub WebSocket connected — subscribing %d symbols", len(symbols))
        for sym in symbols:
            ws.send(json.dumps({"type": "subscribe", "symbol": sym}))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
            if msg.get("type") != "trade" or not msg.get("data"):
                return
            for trade in msg["data"]:
                sym = trade.get("s")
                price = trade.get("p")
                volume = trade.get("v", 0)
                ts = trade.get("t", 0)
                if not sym or price is None:
                    continue
                entry = {"price": float(price), "volume": float(volume), "ts": int(ts)}
                with _lock:
                    _prices[sym] = entry
                _notify(sym, entry)
        except Exception as exc:
            logger.debug("WS message error: %s", exc)

    def on_error(ws, error):
        logger.warning("Finnhub WS error: %s", error)

    def on_close(ws, code, msg):
        global _ws_app
        _ws_app = None
        logger.info("Finnhub WS closed (%s) — reconnecting in 10s", code)
        time.sleep(10)
        _run_ws(api_key, _subscribed_symbols)

    url = f"wss://ws.finnhub.io?token={api_key}"
    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                 on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=30, ping_timeout=10)
