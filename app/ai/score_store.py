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
