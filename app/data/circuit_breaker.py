"""Circuit breaker for data sources (MT5 bridge, Yahoo, Finnhub, …).

Standard 3-state breaker:

    CLOSED ──── failure_rate > threshold ────▶ OPEN
       ▲                                          │
       │                                  reset_timeout elapsed
       │                                          ▼
       └──── probe success ───────────── HALF_OPEN
              probe failure  ──────▶ OPEN (reset)

Why
---
When the MT5 bridge starts returning 502s for 95% of requests, every poll
still hits the dead endpoint. The circuit breaker watches the rolling
failure rate, opens the circuit (short-circuits the call) for a cool-down,
then probes with a single request to detect recovery.

Usage
-----
    cb = CircuitBreaker("mt5", CircuitBreakerConfig(failure_threshold=0.5))
    if not cb.allow_request():
        return serve_stale_cache()
    try:
        data = fetch_from_mt5()
        cb.record_success()
    except BridgeError as exc:
        cb.record_failure()
        return serve_stale_cache()

Node.js porting notes
---------------------
The state machine is portable. The Python implementation uses a `deque` for
the rolling window; in Node you can use a simple `Array.shift()` on each
record. The public API (`allow_request`, `record_success`, `record_failure`,
`state`) is the only contract.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional

logger = logging.getLogger(__name__)

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Tunables for the circuit breaker.

    window_seconds         : rolling window for failure-rate measurement
    min_calls              : ignore the breaker until this many calls in the window
    failure_threshold      : failure rate (0.0-1.0) that opens the circuit
    reset_timeout_seconds  : how long OPEN stays open before HALF_OPEN
    half_open_max_probes   : concurrent probes allowed in HALF_OPEN
    persist_path           : optional file to persist state across restarts
    """
    window_seconds: int = 30
    min_calls: int = 5
    failure_threshold: float = 0.5
    reset_timeout_seconds: int = 60
    half_open_max_probes: int = 1
    persist_path: Optional[str] = None


@dataclass
class _Call:
    ts: float
    success: bool


class CircuitBreaker:
    """Thread-safe circuit breaker with rolling window + half-open probing."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()
        self._calls: Deque[_Call] = deque()
        self._state: str = STATE_CLOSED
        self._opened_at: float = 0.0
        self._probes_in_flight: int = 0
        self._load_from_disk()

    # ── Public API ──────────────────────────────────────────────────────────

    def allow_request(self) -> bool:
        """Returns True if a real call should be made; False to short-circuit."""
        with self._lock:
            self._gc()
            if self._state == STATE_CLOSED:
                return True
            if self._state == STATE_OPEN:
                # Check if reset timeout has elapsed → move to HALF_OPEN
                if (time.time() - self._opened_at) >= self.config.reset_timeout_seconds:
                    self._state = STATE_HALF_OPEN
                    self._probes_in_flight = 0
                    logger.info("[%s] CircuitBreaker → HALF_OPEN (probe)", self.name)
                else:
                    return False
            # HALF_OPEN: allow up to N concurrent probes
            if self._probes_in_flight < self.config.half_open_max_probes:
                self._probes_in_flight += 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._calls.append(_Call(time.time(), True))
            self._gc()
            if self._state == STATE_HALF_OPEN:
                self._state = STATE_CLOSED
                self._probes_in_flight = 0
                self._calls.clear()
                logger.info("[%s] CircuitBreaker → CLOSED (probe ok)", self.name)
                self._persist()
                return
            # Re-evaluate the rate after every call — a 3-fail-then-1-success
            # pattern must still trip the breaker.
            self._maybe_open()
            self._persist()

    def record_failure(self) -> None:
        with self._lock:
            now = time.time()
            self._calls.append(_Call(now, False))
            self._gc()
            if self._state == STATE_HALF_OPEN:
                # Probe failed → reopen
                self._state = STATE_OPEN
                self._opened_at = now
                self._probes_in_flight = 0
                logger.warning("[%s] CircuitBreaker → OPEN (probe failed)", self.name)
                self._persist()
                return

            self._maybe_open()
            self._persist()

    def _maybe_open(self) -> None:
        """Trip the breaker if failure rate crosses threshold in CLOSED state."""
        if self._state != STATE_CLOSED:
            return
        if len(self._calls) < self.config.min_calls:
            return
        rate = self._failure_rate()
        if rate >= self.config.failure_threshold:
            self._state = STATE_OPEN
            self._opened_at = time.time()
            logger.warning(
                "[%s] CircuitBreaker → OPEN (failure_rate=%.0f%% over %d calls)",
                self.name, rate * 100, len(self._calls),
            )

    def state(self) -> str:
        with self._lock:
            self._gc()
            return self._state

    def stats(self) -> dict:
        """Diagnostic snapshot for telemetry."""
        with self._lock:
            self._gc()
            total = len(self._calls)
            failed = sum(1 for c in self._calls if not c.success)
            rate = (failed / total) if total else 0.0
            return {
                "name": self.name,
                "state": self._state,
                "calls_in_window": total,
                "failures_in_window": failed,
                "failure_rate": round(rate, 3),
                "threshold": self.config.failure_threshold,
                "min_calls": self.config.min_calls,
                "window_seconds": self.config.window_seconds,
                "reset_timeout_seconds": self.config.reset_timeout_seconds,
                "opened_at": self._opened_at or None,
            }

    def reset(self) -> None:
        """Force the breaker back to CLOSED."""
        with self._lock:
            self._state = STATE_CLOSED
            self._calls.clear()
            self._opened_at = 0.0
            self._probes_in_flight = 0
            self._persist()
            logger.info("[%s] CircuitBreaker → CLOSED (manual reset)", self.name)

    # ── Internals ───────────────────────────────────────────────────────────

    def _failure_rate(self) -> float:
        if not self._calls:
            return 0.0
        failed = sum(1 for c in self._calls if not c.success)
        return failed / len(self._calls)

    def _gc(self) -> None:
        """Drop calls outside the rolling window."""
        cutoff = time.time() - self.config.window_seconds
        while self._calls and self._calls[0].ts < cutoff:
            self._calls.popleft()

    def _persist(self) -> None:
        path = self.config.persist_path
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "state": self._state,
                "opened_at": self._opened_at,
                "calls": [{"ts": c.ts, "success": c.success} for c in self._calls],
            }
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("CircuitBreaker persist failed: %s", exc)

    def _load_from_disk(self) -> None:
        path = self.config.persist_path
        if not path or not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._state = str(data.get("state", STATE_CLOSED))
            self._opened_at = float(data.get("opened_at", 0))
            self._calls = deque(
                _Call(float(c["ts"]), bool(c["success"]))
                for c in data.get("calls", [])
            )
        except Exception as exc:
            logger.debug("CircuitBreaker load failed: %s", exc)


# ── Module-level registry (one breaker per source) ──────────────────────────

_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_or_create(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Return a singleton breaker for `name` (e.g. 'mt5', 'yahoo')."""
    with _registry_lock:
        cb = _registry.get(name)
        if cb is None:
            cb = CircuitBreaker(name, config)
            _registry[name] = cb
        return cb


def all_stats() -> list[dict]:
    with _registry_lock:
        return [cb.stats() for cb in _registry.values()]
