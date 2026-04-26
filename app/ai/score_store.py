"""Thread-safe store for AI ticker scores with TTL expiry."""
import time
from threading import Lock

# ticker -> (score, rationale, sources, timestamp)
_store: dict[str, tuple[float, str, list, float]] = {}
_lock = Lock()


def set_score(ticker: str, score: float, rationale: str = "", sources: list | None = None) -> None:
    with _lock:
        _store[ticker] = (score, rationale, sources or [], time.time())


def get_entry(ticker: str, ttl: int = 3600) -> dict | None:
    """Return {score, rationale, sources, ts} or None if missing/expired."""
    with _lock:
        entry = _store.get(ticker)
        if entry is None:
            return None
        score, rationale, sources, ts = entry
        if (time.time() - ts) >= ttl:
            return None
        return {"score": score, "rationale": rationale, "sources": sources, "ts": ts}


def get_score(ticker: str, ttl: int = 3600) -> float | None:
    entry = get_entry(ticker, ttl)
    return entry["score"] if entry else None


def get_all_scores(ttl: int = 3600) -> dict[str, float]:
    now = time.time()
    with _lock:
        return {t: s for t, (s, _, _src, ts) in _store.items() if (now - ts) < ttl}


def clear() -> None:
    with _lock:
        _store.clear()


# ── Autonomous decision store ────────────────────────────────────────────────
# ticker -> (action, confidence, position_size_pct, stop_loss_pct,
#            take_profit_pct, time_horizon, rationale, key_risks, sources, ts)
_decision_store: dict[str, tuple] = {}
_decision_lock = Lock()


def set_decision(
    ticker: str, action: str, confidence: float,
    position_size_pct: float, stop_loss_pct: float, take_profit_pct: float,
    time_horizon: str, rationale: str, key_risks: str, sources: list,
) -> None:
    with _decision_lock:
        _decision_store[ticker] = (
            action, confidence, position_size_pct, stop_loss_pct,
            take_profit_pct, time_horizon, rationale, key_risks, sources, time.time(),
        )


def get_decision(ticker: str, ttl: int = 7200) -> dict | None:
    with _decision_lock:
        entry = _decision_store.get(ticker)
        if entry is None:
            return None
        action, confidence, pos, sl, tp, horizon, rationale, risks, sources, ts = entry
        if (time.time() - ts) >= ttl:
            return None
        return {
            "action": action, "confidence": confidence,
            "position_size_pct": pos, "stop_loss_pct": sl,
            "take_profit_pct": tp, "time_horizon": horizon,
            "rationale": rationale, "key_risks": risks,
            "sources": sources, "ts": ts,
        }


def get_all_decisions(ttl: int = 7200) -> dict[str, dict]:
    now = time.time()
    with _decision_lock:
        result = {}
        for ticker, entry in _decision_store.items():
            action, confidence, pos, sl, tp, horizon, rationale, risks, sources, ts = entry
            if (now - ts) < ttl:
                result[ticker] = {
                    "action": action, "confidence": confidence,
                    "position_size_pct": pos, "stop_loss_pct": sl,
                    "take_profit_pct": tp, "time_horizon": horizon,
                    "rationale": rationale, "key_risks": risks,
                    "sources": sources, "ts": ts,
                }
        return result
