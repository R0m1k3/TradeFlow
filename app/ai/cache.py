"""Persistent file-based cache for AI responses.

Avoids re-querying the LLM provider for the same (provider, model, prompt_hash)
tuple within a TTL window. Survives process restarts. Thread-safe.

Layout: data/ai_cache/<key>.json  →  {"value": ..., "ts": float}
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ai_cache"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_in_mem: dict[str, tuple[float, str]] = {}  # key -> (ts, value_json)
_MAX_IN_MEM = 512


def _key(provider: str, model: str, prompt: str, mode: str = "") -> str:
    """Stable cache key. mode distinguishes hybrid vs autonomous prompts."""
    raw = f"{provider}|{model}|{mode}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get(provider: str, model: str, prompt: str, ttl: int, mode: str = "") -> Optional[Any]:
    """Return cached value if present and not expired, else None."""
    key = _key(provider, model, prompt, mode)
    now = time.time()

    with _lock:
        # 1) in-memory fast path
        cached = _in_mem.get(key)
        if cached is not None:
            ts, value_json = cached
            if (now - ts) < ttl:
                try:
                    return json.loads(value_json)
                except Exception:
                    pass
            else:
                _in_mem.pop(key, None)

        # 2) disk fallback
        path = _DATA_DIR / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ts = float(data.get("ts", 0))
                if (now - ts) < ttl:
                    if len(_in_mem) >= _MAX_IN_MEM:
                        # tiny LRU: drop oldest
                        oldest = min(_in_mem.items(), key=lambda kv: kv[1][0])[0]
                        _in_mem.pop(oldest, None)
                    _in_mem[key] = (ts, data.get("value", "null"))
                    return json.loads(data.get("value", "null"))
                # expired — clean up
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("Cache read error for %s: %s", key, exc)
    return None


def set(provider: str, model: str, prompt: str, value: Any, mode: str = "") -> None:
    """Persist a successful AI response."""
    key = _key(provider, model, prompt, mode)
    ts = time.time()
    try:
        value_json = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        value_json = json.dumps(str(value))
    with _lock:
        if len(_in_mem) >= _MAX_IN_MEM:
            oldest = min(_in_mem.items(), key=lambda kv: kv[1][0])[0]
            _in_mem.pop(oldest, None)
        _in_mem[key] = (ts, value_json)
        try:
            (_DATA_DIR / f"{key}.json").write_text(
                json.dumps({"ts": ts, "value": value_json}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Cache write error for %s: %s", key, exc)


def clear() -> int:
    """Drop all cached entries. Returns number removed."""
    with _lock:
        n = len(_in_mem)
        _in_mem.clear()
    removed = n
    for p in _DATA_DIR.glob("*.json"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def stats() -> dict:
    """Quick stats for diagnostics."""
    with _lock:
        in_mem = len(_in_mem)
    disk = sum(1 for _ in _DATA_DIR.glob("*.json"))
    return {"in_memory": in_mem, "on_disk": disk, "cache_dir": str(_DATA_DIR)}
