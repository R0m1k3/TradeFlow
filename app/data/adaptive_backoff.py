"""Per-key adaptive backoff: stretch retry intervals for persistently failing keys.

Why
---
Some tickers (e.g. `ROG.SW`) systematically 404 from Yahoo, and some bridges
(MT5) return 502 for an entire market. Hammering them every tick is wasteful
and pollutes logs. This module remembers "the last time we tried X for reason
R" and refuses to retry until the appropriate cool-down has elapsed.

Different failure kinds get different backoff curves:
  - 404 / not found       → long, near-static backoff (1 retry / 10 min)
  - 5xx / server error    → exponential, capped (30s → 60s → 120s → …)
  - timeout / network     → exponential, capped
  - 429 / rate limited    → exponential starting at 60s, respecting Retry-After
  - other                 → mild exponential, capped

Usage
-----
    backoff = AdaptiveBackoff()
    if not backoff.should_retry("yahoo:ROG.SW", kind="404"):
        return serve_stale_cache()
    try:
        data = fetch(...)
        backoff.record_success("yahoo:ROG.SW")
    except HttpError as exc:
        backoff.record_failure("yahoo:ROG.SW", kind=exc.kind)
        return serve_stale_cache()

Node.js porting notes
---------------------
The state is a `Map<key, {failures, nextRetryAt}>`. The public surface
(`should_retry`, `record_failure`, `record_success`, `next_retry_at`,
`stats`) maps 1-to-1 to a Node implementation.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Failure kinds — used to pick the backoff curve
KIND_404 = "404"
KIND_5XX = "5xx"
KIND_TIMEOUT = "timeout"
KIND_429 = "429"
KIND_OTHER = "other"


@dataclass
class _Record:
    failures: int = 0
    next_retry_at: float = 0.0
    last_kind: str = KIND_OTHER
    last_attempt: float = 0.0


@dataclass
class AdaptiveBackoffConfig:
    """Tunables for the per-key backoff.

    base_seconds          : initial cool-down after the first failure
    max_seconds           : cap on the cool-down
    factor                : exponential multiplier per consecutive failure
    static_intervals      : for 404s — list of (failure_count, seconds) pairs
    """
    base_seconds: float = 30.0
    max_seconds: float = 600.0      # 10 min cap
    factor: float = 2.0
    # For 404s, use a near-static schedule (the resource is gone, no point retrying fast)
    static_intervals: tuple[tuple[int, float], ...] = (
        (1, 300.0),    # 1st failure → wait 5 min
        (2, 600.0),    # 2nd          → 10 min
        (3, 900.0),    # 3rd+         → 15 min (capped)
    )
    persist_path: Optional[str] = None


class AdaptiveBackoff:
    """Thread-safe per-key backoff with kind-aware curves."""

    def __init__(self, config: AdaptiveBackoffConfig | None = None, name: str = "default") -> None:
        self.config = config or AdaptiveBackoffConfig()
        self._name = name
        self._lock = threading.Lock()
        self._records: dict[str, _Record] = {}
        self._load_from_disk()

    # ── Public API ──────────────────────────────────────────────────────────

    def should_retry(self, key: str, kind: str = KIND_OTHER) -> bool:
        """True if a real call is allowed now (or has never been tried)."""
        rec = self._peek(key)
        if rec is None or rec.failures == 0:
            return True
        return time.time() >= rec.next_retry_at

    def record_failure(self, key: str, kind: str = KIND_OTHER) -> float:
        """Record a failure. Returns the new `next_retry_at` (epoch seconds)."""
        now = time.time()
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                rec = _Record()
                self._records[key] = rec
            rec.failures += 1
            rec.last_kind = kind
            rec.last_attempt = now
            rec.next_retry_at = now + self._compute_delay(rec.failures, kind)
            self._persist()
            delay = rec.next_retry_at - now
            logger.info(
                "[%s] Backoff %s → next attempt in %.1fs (failure #%d, kind=%s)",
                self._name, key, delay, rec.failures, kind,
            )
            return rec.next_retry_at

    def record_success(self, key: str) -> None:
        """Clear any backoff state for `key`."""
        with self._lock:
            if self._records.pop(key, None) is not None:
                self._persist()

    def next_retry_at(self, key: str) -> float | None:
        rec = self._peek(key)
        if rec is None or rec.failures == 0:
            return None
        return rec.next_retry_at

    def status(self, key: str) -> dict:
        rec = self._peek(key)
        if rec is None or rec.failures == 0:
            return {"failures": 0, "next_retry_at": None, "last_kind": None}
        return {
            "failures": rec.failures,
            "next_retry_at": rec.next_retry_at,
            "last_kind": rec.last_kind,
        }

    def reset(self, key: str | None = None) -> int:
        with self._lock:
            if key is None:
                n = len(self._records)
                self._records.clear()
            else:
                n = 1 if self._records.pop(key, None) is not None else 0
            self._persist()
            return n

    def stats(self) -> dict:
        now = time.time()
        active = blocked = 0
        with self._lock:
            for rec in self._records.values():
                if rec.failures == 0:
                    continue
                active += 1
                if now < rec.next_retry_at:
                    blocked += 1
        return {
            "name": self._name,
            "tracked_keys": len(self._records),
            "active": active,
            "blocked": blocked,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _compute_delay(self, failures: int, kind: str) -> float:
        if kind == KIND_404:
            # Near-static schedule for permanent errors
            for thr, seconds in self.config.static_intervals:
                if failures <= thr:
                    return seconds
            return self.config.static_intervals[-1][1]
        if kind == KIND_429:
            # Treat 429 like 5xx but with a bigger base
            base = max(self.config.base_seconds, 60.0)
        else:
            base = self.config.base_seconds
        delay = base * (self.config.factor ** (failures - 1))
        return min(delay, self.config.max_seconds)

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
                    "next_retry_at": r.next_retry_at,
                    "last_kind": r.last_kind,
                    "last_attempt": r.last_attempt,
                }
                for k, r in self._records.items()
            }
            p.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("AdaptiveBackoff persist failed: %s", exc)

    def _load_from_disk(self) -> None:
        path = self.config.persist_path
        if not path or not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for k, v in data.items():
                self._records[k] = _Record(
                    failures=int(v.get("failures", 0)),
                    next_retry_at=float(v.get("next_retry_at", 0)),
                    last_kind=str(v.get("last_kind", KIND_OTHER)),
                    last_attempt=float(v.get("last_attempt", 0)),
                )
        except Exception as exc:
            logger.debug("AdaptiveBackoff load failed: %s", exc)
