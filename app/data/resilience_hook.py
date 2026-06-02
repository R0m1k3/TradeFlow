"""High-level integration: combine the 3 resilience primitives behind one API.

This is the *only* file the rest of the codebase needs to import. It wires
together:

  * CircuitBreaker  → "is the upstream source alive at all?"
  * NegativeCache   → "is THIS specific key dead?"
  * AdaptiveBackoff → "when should I retry this key?"

Public API
----------
    guard = ResilienceGuard.for_source("mt5")
    decision = guard.before_call("ROG.SW:/ohlcv")  # "call" | "stale"
    if decision == "stale":
        return serve_cached(...)
    try:
        data = call_upstream(...)
        guard.after_success("ROG.SW:/ohlcv")
    except Http404:
        guard.after_failure("ROG.SW:/ohlcv", kind="404")
    except Http502:
        guard.after_failure("ROG.SW:/ohlcv", kind="5xx")
    ...

Node.js porting notes
---------------------
This whole module is pure orchestration — port the file structure and the
`for_source(name) → guard` registry, and the rest is mechanical.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from app.data.adaptive_backoff import (
    AdaptiveBackoff,
    AdaptiveBackoffConfig,
    KIND_404,
    KIND_429,
    KIND_5XX,
    KIND_OTHER,
    KIND_TIMEOUT,
)
from app.data.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
)
from app.data.negative_cache import (
    KIND_5XX as _NC_5XX,  # re-export
    KIND_404 as _NC_404,
    KIND_TIMEOUT as _NC_TIMEOUT,
    KIND_OTHER as _NC_OTHER,
    NegativeCache,
    NegativeCacheConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ResilienceGuard",
    "Decision",
    "KIND_404",
    "KIND_5XX",
    "KIND_TIMEOUT",
    "KIND_429",
    "KIND_OTHER",
]


@dataclass
class Decision:
    """Result of `guard.before_call(key)`."""
    proceed: bool           # True = make the real call; False = serve stale
    reason: str             # human-readable explanation
    state: str              # "normal" | "skip" | "open" | "degraded" | "half_open"


class ResilienceGuard:
    """All-in-one resilience wrapper for a single upstream source."""

    def __init__(
        self,
        name: str,
        *,
        breaker: CircuitBreaker | None = None,
        negative: NegativeCache | None = None,
        backoff: AdaptiveBackoff | None = None,
    ) -> None:
        self.name = name
        self.breaker = breaker or CircuitBreaker(name)
        self.negative = negative or NegativeCache(name=name)
        self.backoff = backoff or AdaptiveBackoff(name=name)

    # ── Call lifecycle ─────────────────────────────────────────────────────

    def before_call(self, key: str) -> Decision:
        """Return whether the caller should make a real upstream request."""
        # 1. Hardest gate: the breaker is open → don't even consider
        if not self.breaker.allow_request():
            return Decision(False, "breaker open", self.breaker.state())

        # 2. Per-key backoff window not yet elapsed
        if not self.backoff.should_retry(key):
            return Decision(False, f"backoff active for {key}", "degraded")

        # 3. Negative cache says this key is in skip mode
        if self.negative.should_skip(key):
            return Decision(False, f"negative-cache skip for {key}", "skip")

        return Decision(True, "ok", "normal")

    def after_success(self, key: str) -> None:
        """Tell all 3 layers the call succeeded."""
        self.breaker.record_success()
        self.negative.record_success(key)
        self.backoff.record_success(key)

    def after_failure(self, key: str, kind: str = KIND_OTHER) -> None:
        """Tell all 3 layers the call failed; pick the right kind for each."""
        # Map to negative-cache kind vocabulary
        nc_kind = {
            KIND_404: _NC_404,
            KIND_5XX: _NC_5XX,
            KIND_TIMEOUT: _NC_TIMEOUT,
        }.get(kind, _NC_OTHER)

        self.breaker.record_failure()
        self.negative.record(key, nc_kind)
        self.backoff.record_failure(key, kind)

    # ── Telemetry ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "name": self.name,
            "breaker": self.breaker.stats(),
            "negative_cache": self.negative.stats(),
            "backoff": self.backoff.stats(),
        }

    def reset(self) -> None:
        self.breaker.reset()
        self.negative.reset()
        self.backoff.reset()


# ── Source registry (one guard per upstream) ───────────────────────────────

_registry: dict[str, ResilienceGuard] = {}
_lock = threading.Lock()

# Pre-tuned defaults for known sources
_PRESETS: dict[str, dict] = {
    # MT5 bridge: aggressive — open the circuit fast, cool down for a minute
    "mt5": dict(
        breaker=CircuitBreakerConfig(
            window_seconds=30, min_calls=5, failure_threshold=0.5,
            reset_timeout_seconds=60, half_open_max_probes=1,
        ),
        negative=NegativeCacheConfig(
            failure_threshold=3, window_seconds=600,
            skip_duration=120, max_skip_duration=900,
        ),
        backoff=AdaptiveBackoffConfig(
            base_seconds=30, max_seconds=300, factor=2.0,
        ),
    ),
    # Yahoo: gentle — 404s are permanent, 5xx usually transient
    "yahoo": dict(
        breaker=CircuitBreakerConfig(
            window_seconds=60, min_calls=10, failure_threshold=0.6,
            reset_timeout_seconds=120, half_open_max_probes=2,
        ),
        negative=NegativeCacheConfig(
            failure_threshold=3, window_seconds=600,
            skip_duration=300, max_skip_duration=1800,
        ),
        backoff=AdaptiveBackoffConfig(
            base_seconds=60, max_seconds=900, factor=2.0,
        ),
    ),
    # Finnhub: rate-limited, treat 429 specially
    "finnhub": dict(
        breaker=CircuitBreakerConfig(
            window_seconds=30, min_calls=5, failure_threshold=0.5,
            reset_timeout_seconds=60, half_open_max_probes=1,
        ),
        negative=NegativeCacheConfig(
            failure_threshold=3, window_seconds=300,
            skip_duration=60, max_skip_duration=300,
        ),
        backoff=AdaptiveBackoffConfig(
            base_seconds=60, max_seconds=600, factor=2.0,
        ),
    ),
}


def for_source(name: str) -> ResilienceGuard:
    """Return (or create) the singleton guard for the given upstream source.

    `name` examples: "mt5", "yahoo", "finnhub", "openrouter".
    """
    with _lock:
        guard = _registry.get(name)
        if guard is not None:
            return guard
        preset = _PRESETS.get(name, {})
        guard = ResilienceGuard(
            name,
            breaker=preset.get("breaker") and CircuitBreaker(name, preset["breaker"]),
            negative=preset.get("negative") and NegativeCache(name=name, config=preset["negative"]),
            backoff=preset.get("backoff") and AdaptiveBackoff(name=name, config=preset["backoff"]),
        )
        _registry[name] = guard
        return guard


def all_stats() -> list[dict]:
    """Snapshot of every registered guard — for the /api/admin/resilience endpoint."""
    with _lock:
        return [g.stats() for g in _registry.values()]


# ── Convenience: decorator / context manager ────────────────────────────────


class resilient:
    """Context manager that wraps a single upstream call.

    Example:
        with resilient("mt5", f"{symbol}:/ohlcv") as r:
            if not r.should_call:
                return serve_stale(...)
            try:
                data = upstream_call()
            except HttpError as e:
                r.fail(kind=classify(e))
                return serve_stale(...)
            r.ok()
            return data
    """

    def __init__(self, source: str, key: str) -> None:
        self.guard = for_source(source)
        self.key = key
        self.should_call = False
        self._state = "normal"

    def __enter__(self) -> "resilient":
        d = self.guard.before_call(self.key)
        self.should_call = d.proceed
        self._state = d.state
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            kind = classify_exception(exc)
            self.guard.after_failure(self.key, kind)

    def ok(self) -> None:
        self.guard.after_success(self.key)

    def fail(self, kind: str = KIND_OTHER) -> None:
        self.guard.after_failure(self.key, kind)


def classify_exception(exc: BaseException) -> str:
    """Best-effort mapping of an exception to a failure kind string."""
    msg = (str(exc) or "").lower()
    if "404" in msg or "not found" in msg:
        return KIND_404
    if "429" in msg or "rate" in msg:
        return KIND_429
    if "timeout" in msg or "timed out" in msg:
        return KIND_TIMEOUT
    if any(c in msg for c in ("500", "502", "503", "504", "bad gateway", "service unavailable")):
        return KIND_5XX
    return KIND_OTHER
