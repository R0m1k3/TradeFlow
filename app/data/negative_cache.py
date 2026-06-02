"""Per-key negative cache: temporarily skip data sources that keep failing.

Use case
--------
When a data source (MT5 bridge, Yahoo, etc.) starts returning 502/404 for a
specific ticker (e.g. `ROG.SW` 404 from Yahoo), we don't want every poll to
re-hit the dead endpoint. This module records each failure and, after a
threshold, tells the caller to short-circuit and serve stale cache.

Design
------
* In-memory, thread-safe, optional disk persistence.
* Per key (e.g. "mt5:ROG.SW:/ohlcv"), with rolling failure window.
* Three behaviours:
    - NORMAL  → let the call through
    - DEGRADED → let the call through but record every failure (we may recover)
    - SKIP    → reject the call, force caller to serve stale cache
* State transitions are time-based — failures age out after `window_seconds`.

Node.js porting notes
---------------------
The state machine is the same in any language. The only API surface is:
    record(key, kind), status(key) -> {"state", "failures", "skip_until"}
    should_skip(key) -> bool, reset(key) -> None
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Failure kind hints — used for telemetry & adaptive backoff
KIND_404 = "404"          # not found, almost certainly permanent
KIND_5XX = "5xx"          # server-side problem, may recover
KIND_TIMEOUT = "timeout"  # network/bridge issue
KIND_OTHER = "other"


@dataclass
class _Record:
    """Per-key failure state."""
    failures: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    last_kind: str = KIND_OTHER
    skip_until: float = 0.0  # epoch seconds — caller should skip while now < skip_until


@dataclass
class NegativeCacheConfig:
    """Tunables for the negative cache.

    failure_threshold  : consecutive failures inside the window before SKIP
    window_seconds     : rolling window; old failures decay away
    skip_duration      : how long to stay in SKIP after threshold is reached
    max_skip_duration  : cap on dynamic backoff (cumulative failures extend it)
    persist_path       : optional file to persist state across restarts
    """
    failure_threshold: int = 3
    window_seconds: int = 600
    skip_duration: int = 300          # 5 min
    max_skip_duration: int = 1800     # 30 min
    persist_path: Optional[str] = None


class NegativeCache:
    """Thread-safe per-key negative cache with optional disk persistence."""

    def __init__(self, config: NegativeCacheConfig | None = None, name: str = "default") -> None:
        self.config = config or NegativeCacheConfig()
        self._name = name
        self._lock = threading.Lock()
        self._records: dict[str, _Record] = {}
        self._load_from_disk()

    # ── Public API ──────────────────────────────────────────────────────────

    def record(self, key: str, kind: str = KIND_OTHER) -> str:
        """Record a failure for `key`. Returns the new state."""
        now = time.time()
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                rec = _Record()
                self._records[key] = rec
            # Decay: if last failure is outside the window, reset
            if rec.last_ts and (now - rec.last_ts) > self.config.window_seconds:
                rec.failures = 0
                rec.first_ts = now
            rec.failures += 1
            rec.last_ts = now
            rec.last_kind = kind

            if rec.failures >= self.config.failure_threshold:
                # Capped exponential skip: 1×, 2×, 4×, ... up to max_skip_duration
                over = rec.failures - self.config.failure_threshold
                duration = min(
                    self.config.skip_duration * (2 ** over),
                    self.config.max_skip_duration,
                )
                rec.skip_until = now + duration
                self._persist()
                logger.warning(
                    "[%s] NegativeCache %s → SKIP for %.0fs after %d failures (last=%s)",
                    self._name, key, duration, rec.failures, kind,
                )
                return "skip"
            self._persist()
            return "degraded"

    def record_success(self, key: str) -> None:
        """Clear any failure state for `key`."""
        with self._lock:
            if self._records.pop(key, None) is not None:
                self._persist()

    def should_skip(self, key: str) -> bool:
        """True if the caller should short-circuit and serve stale data."""
        rec = self._peek(key)
        if rec is None or rec.skip_until == 0.0:
            return False
        return time.time() < rec.skip_until

    def status(self, key: str) -> dict:
        """Diagnostic snapshot for telemetry / dashboards."""
        rec = self._peek(key)
        if rec is None:
            return {"state": "normal", "failures": 0, "skip_until": None, "last_kind": None}
        now = time.time()
        if rec.skip_until and now < rec.skip_until:
            state = "skip"
        elif rec.failures > 0:
            state = "degraded"
        else:
            state = "normal"
        return {
            "state": state,
            "failures": rec.failures,
            "skip_until": rec.skip_until if rec.skip_until else None,
            "last_kind": rec.last_kind,
        }

    def reset(self, key: str | None = None) -> int:
        """Clear state for one key (or all). Returns number cleared."""
        with self._lock:
            if key is None:
                n = len(self._records)
                self._records.clear()
            else:
                n = 1 if self._records.pop(key, None) is not None else 0
            self._persist()
            return n

    def stats(self) -> dict:
        """Aggregate counts for diagnostics."""
        now = time.time()
        skipping = degraded = 0
        with self._lock:
            for rec in self._records.values():
                if rec.skip_until and now < rec.skip_until:
                    skipping += 1
                elif rec.failures > 0:
                    degraded += 1
        return {
            "name": self._name,
            "tracked_keys": len(self._records),
            "skipping": skipping,
            "degraded": degraded,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "window_seconds": self.config.window_seconds,
                "skip_duration": self.config.skip_duration,
                "max_skip_duration": self.config.max_skip_duration,
            },
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _peek(self, key: str) -> _Record | None:
        with self._lock:
            return self._records.get(key)

    def _persist(self) -> None:
        path = self.config.persist_path
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            snapshot = {
                k: {
                    "failures": r.failures,
                    "first_ts": r.first_ts,
                    "last_ts": r.last_ts,
                    "last_kind": r.last_kind,
                    "skip_until": r.skip_until,
                }
                for k, r in self._records.items()
            }
            p.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("NegativeCache persist failed: %s", exc)

    def _load_from_disk(self) -> None:
        path = self.config.persist_path
        if not path or not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for k, v in data.items():
                self._records[k] = _Record(
                    failures=int(v.get("failures", 0)),
                    first_ts=float(v.get("first_ts", 0)),
                    last_ts=float(v.get("last_ts", 0)),
                    last_kind=str(v.get("last_kind", KIND_OTHER)),
                    skip_until=float(v.get("skip_until", 0)),
                )
        except Exception as exc:
            logger.debug("NegativeCache load failed: %s", exc)


# ── Module-level default cache (for simple use) ─────────────────────────────

_default: NegativeCache | None = None


def get_default() -> NegativeCache:
    global _default
    if _default is None:
        _default = NegativeCache(name="default")
    return _default
