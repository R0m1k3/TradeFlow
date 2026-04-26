"""ARIA virtual portfolio — tracks autonomous agent's trading performance."""
import json
import time
import threading
from pathlib import Path

_FILE = Path(__file__).resolve().parents[2] / "data" / "aria_portfolio.json"
_lock = threading.Lock()

_DEFAULT: dict = {
    "initial_capital": 10000.0,
    "cash": 10000.0,
    "positions": {},   # symbol -> {entry_price, quantity, entry_ts, stop_loss_pct, take_profit_pct, invested}
    "trades": [],      # closed trade records
    "snapshots": [],   # {ts, value} for chart
}


def _load() -> dict:
    if _FILE.exists():
        try:
            d = json.loads(_FILE.read_text())
            # Back-compat: ensure all keys exist
            for k, v in _DEFAULT.items():
                d.setdefault(k, v)
            return d
        except Exception:
            pass
    return dict(_DEFAULT)


def _save(p: dict) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(p, indent=2))


def _total_value(p: dict, prices: dict[str, float]) -> float:
    value = p["cash"]
    for sym, pos in p["positions"].items():
        price = prices.get(sym, pos["entry_price"])
        value += price * pos["quantity"]
    return value


# ── Public API ────────────────────────────────────────────────────────────────

def set_initial_capital(amount: float) -> None:
    """Set initial capital only if the portfolio has never traded."""
    with _lock:
        p = _load()
        if not p["trades"] and not p["positions"]:
            p["initial_capital"] = amount
            p["cash"] = amount
            _save(p)


def get_open_positions() -> list[str]:
    with _lock:
        return list(_load()["positions"].keys())


def execute_decision(
    symbol: str,
    action: str,
    position_size_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    current_price: float,
    commission: float = 0.001,
) -> dict | None:
    """Execute BUY or SELL against ARIA's virtual portfolio. Returns trade record or None."""
    if current_price <= 0:
        return None

    with _lock:
        p = _load()
        result = None

        if action == "BUY" and symbol not in p["positions"]:
            amount = (position_size_pct / 100) * p["cash"]
            if amount < 5:
                return None
            fee = amount * commission
            quantity = (amount - fee) / current_price
            p["positions"][symbol] = {
                "entry_price": current_price,
                "quantity": quantity,
                "entry_ts": time.time(),
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "invested": amount,
            }
            p["cash"] -= amount
            result = {
                "action": "BUY", "symbol": symbol,
                "price": current_price, "quantity": quantity,
                "fee": fee, "ts": time.time(),
            }

        elif action == "SELL" and symbol in p["positions"]:
            pos = p["positions"].pop(symbol)
            proceeds = pos["quantity"] * current_price
            fee = proceeds * commission
            net_proceeds = proceeds - fee
            pnl = net_proceeds - pos["invested"]
            p["cash"] += net_proceeds
            result = {
                "action": "SELL", "symbol": symbol,
                "entry_price": pos["entry_price"], "exit_price": current_price,
                "quantity": pos["quantity"], "invested": pos["invested"],
                "pnl": pnl, "pnl_pct": (pnl / pos["invested"]) * 100,
                "fee": fee, "ts": time.time(),
            }
            p["trades"].append(result)

        if result:
            _save(p)
        return result


def check_stops(prices: dict[str, float], commission: float = 0.001) -> list[dict]:
    """Auto-close positions that hit their stop-loss or take-profit levels."""
    triggered = []
    with _lock:
        p = _load()
        to_close = []
        for sym, pos in p["positions"].items():
            price = prices.get(sym)
            if price is None:
                continue
            change_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
            if change_pct <= -pos["stop_loss_pct"]:
                to_close.append((sym, price, "SL"))
            elif change_pct >= pos["take_profit_pct"]:
                to_close.append((sym, price, "TP"))

        for sym, price, reason in to_close:
            pos = p["positions"].pop(sym)
            proceeds = pos["quantity"] * price
            fee = proceeds * commission
            net_proceeds = proceeds - fee
            pnl = net_proceeds - pos["invested"]
            p["cash"] += net_proceeds
            trade = {
                "action": "SELL", "symbol": sym, "reason": reason,
                "entry_price": pos["entry_price"], "exit_price": price,
                "quantity": pos["quantity"], "invested": pos["invested"],
                "pnl": pnl, "pnl_pct": (pnl / pos["invested"]) * 100,
                "fee": fee, "ts": time.time(),
            }
            p["trades"].append(trade)
            triggered.append(trade)

        if triggered:
            _save(p)
    return triggered


def take_snapshot(prices: dict[str, float] | None = None) -> float:
    """Record current portfolio value. Returns the value."""
    with _lock:
        p = _load()
        value = _total_value(p, prices or {})
        p["snapshots"].append({"ts": time.time(), "value": value})
        if len(p["snapshots"]) > 2000:
            p["snapshots"] = p["snapshots"][-2000:]
        _save(p)
        return value


def get_stats(prices: dict[str, float] | None = None) -> dict:
    """Return full portfolio stats for comparison."""
    with _lock:
        p = _load()
        px = prices or {}
        current = _total_value(p, px)
        initial = p["initial_capital"]
        pnl = current - initial
        trades = p["trades"]
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        return {
            "initial_capital": initial,
            "cash": p["cash"],
            "current_value": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round((pnl / initial * 100) if initial else 0, 2),
            "total_trades": len(trades),
            "open_positions": len(p["positions"]),
            "win_rate": round(len(wins) / len(trades), 3) if trades else 0,
            "positions": p["positions"],
            "recent_trades": trades[-20:][::-1],
            "snapshots": p["snapshots"],
        }


def reset(initial_capital: float) -> None:
    """Hard reset — wipes all trades and positions."""
    with _lock:
        p = {**_DEFAULT, "initial_capital": initial_capital, "cash": initial_capital}
        _save(p)
