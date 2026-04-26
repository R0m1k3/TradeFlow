"""Thread-safe store for AI ticker scores with TTL expiry."""
import time
from threading import Lock

_store: dict[str, tuple[float, float]] = {}  # ticker -> (score, timestamp)
_lock = Lock()


def set_score(ticker: str, score: float) -> None:
    with _lock:
        _store[ticker] = (score, time.time())


def get_score(ticker: str, ttl: int = 3600) -> float | None:
    with _lock:
        entry = _store.get(ticker)
        if entry is None:
            return None
        score, ts = entry
        return score if (time.time() - ts) < ttl else None


def get_all_scores(ttl: int = 3600) -> dict[str, float]:
    now = time.time()
    with _lock:
        return {t: s for t, (s, ts) in _store.items() if (now - ts) < ttl}


def clear() -> None:
    with _lock:
        _store.clear()
