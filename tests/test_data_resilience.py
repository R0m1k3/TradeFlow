"""Tests for the resilience layer (negative cache, circuit breaker, adaptive backoff).

All tests are pure unit tests — no network, no external services.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.data.adaptive_backoff import (
    KIND_404,
    KIND_429,
    KIND_5XX,
    KIND_OTHER,
    KIND_TIMEOUT,
    AdaptiveBackoff,
    AdaptiveBackoffConfig,
)
from app.data.circuit_breaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerConfig,
)
from app.data.negative_cache import (
    KIND_404 as NC_404,
    KIND_5XX as NC_5XX,
    KIND_TIMEOUT as NC_TIMEOUT,
    KIND_OTHER as NC_OTHER,
    NegativeCache,
    NegativeCacheConfig,
)
from app.data.resilience_hook import (
    Decision,
    ResilienceGuard,
    classify_exception,
    for_source,
    resilient,
)


# ── NegativeCache ───────────────────────────────────────────────────────────


class TestNegativeCache:
    def setup_method(self) -> None:
        self.c = NegativeCache(NegativeCacheConfig(failure_threshold=3, window_seconds=60,
                                                   skip_duration=10, max_skip_duration=60))

    def test_initial_state_normal(self):
        assert not self.c.should_skip("ROG.SW")
        assert self.c.status("ROG.SW")["state"] == "normal"

    def test_degraded_before_threshold(self):
        self.c.record("ROG.SW", NC_404)
        self.c.record("ROG.SW", NC_404)
        assert self.c.status("ROG.SW")["state"] == "degraded"
        assert not self.c.should_skip("ROG.SW")

    def test_skip_after_threshold(self):
        for _ in range(3):
            self.c.record("ROG.SW", NC_404)
        assert self.c.should_skip("ROG.SW")
        assert self.c.status("ROG.SW")["state"] == "skip"

    def test_skip_has_deadline(self):
        for _ in range(3):
            self.c.record("ROG.SW", NC_404)
        status = self.c.status("ROG.SW")
        assert status["skip_until"] is not None
        assert status["skip_until"] > time.time()

    def test_success_clears_state(self):
        for _ in range(3):
            self.c.record("ROG.SW", NC_404)
        assert self.c.should_skip("ROG.SW")
        self.c.record_success("ROG.SW")
        assert not self.c.should_skip("ROG.SW")

    def test_window_decay_resets_counter(self):
        c = NegativeCache(NegativeCacheConfig(failure_threshold=3, window_seconds=1,
                                              skip_duration=10, max_skip_duration=60))
        c.record("X", NC_5XX)
        c.record("X", NC_5XX)
        time.sleep(1.1)
        # After the window elapses, next record starts a new window
        c.record("X", NC_5XX)
        assert c.status("X")["state"] == "degraded"
        assert not c.should_skip("X")

    def test_skip_duration_capped(self):
        c = NegativeCache(NegativeCacheConfig(failure_threshold=2, window_seconds=60,
                                              skip_duration=10, max_skip_duration=15))
        c.record("X", NC_5XX)
        c.record("X", NC_5XX)
        # 3rd failure → 10s × 2 = 20s, capped at 15s
        c.record("X", NC_5XX)
        status = c.status("X")
        delta = status["skip_until"] - time.time()
        assert 14 <= delta <= 15.5

    def test_persists_across_instances(self, tmp_path: Path):
        path = str(tmp_path / "nc.json")
        c1 = NegativeCache(NegativeCacheConfig(failure_threshold=2, persist_path=path))
        c1.record("ROG.SW", NC_404)
        c1.record("ROG.SW", NC_404)
        assert c1.should_skip("ROG.SW")

        c2 = NegativeCache(NegativeCacheConfig(failure_threshold=2, persist_path=path))
        assert c2.should_skip("ROG.SW")

    def test_stats(self):
        self.c.record("A", NC_5XX)
        self.c.record("B", NC_5XX)
        self.c.record("B", NC_5XX)
        self.c.record("B", NC_5XX)
        s = self.c.stats()
        assert s["tracked_keys"] == 2
        assert s["degraded"] == 1  # A
        assert s["skipping"] == 1  # B


# ── CircuitBreaker ─────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def setup_method(self) -> None:
        self.cb = CircuitBreaker("test", CircuitBreakerConfig(
            window_seconds=10, min_calls=4, failure_threshold=0.5,
            reset_timeout_seconds=1, half_open_max_probes=1,
        ))

    def test_starts_closed(self):
        assert self.cb.state() == STATE_CLOSED
        assert self.cb.allow_request()

    def test_stays_closed_below_min_calls(self):
        self.cb.record_failure()
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state() == STATE_CLOSED

    def test_opens_when_failure_rate_exceeds(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()  # 3 fail, 1 ok → 75% fail
        assert self.cb.state() == STATE_OPEN
        assert not self.cb.allow_request()

    def test_half_open_after_reset_timeout(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()
        assert self.cb.state() == STATE_OPEN
        time.sleep(1.1)  # reset_timeout
        # Next allow_request should transition to HALF_OPEN and admit a probe
        assert self.cb.allow_request()
        assert self.cb.state() == STATE_HALF_OPEN

    def test_half_open_closes_on_probe_success(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()
        time.sleep(1.1)
        assert self.cb.allow_request()
        self.cb.record_success()
        assert self.cb.state() == STATE_CLOSED

    def test_half_open_reopens_on_probe_failure(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()
        time.sleep(1.1)
        assert self.cb.allow_request()
        self.cb.record_failure()
        assert self.cb.state() == STATE_OPEN

    def test_half_open_limited_probes(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()
        time.sleep(1.1)
        # First probe allowed
        assert self.cb.allow_request()
        # Second probe denied (max=1)
        assert not self.cb.allow_request()

    def test_stats(self):
        self.cb.record_failure()
        s = self.cb.stats()
        assert s["state"] == STATE_CLOSED
        assert s["failures_in_window"] == 1
        assert s["failure_rate"] == 1.0

    def test_reset(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb.record_success()
        assert self.cb.state() == STATE_OPEN
        self.cb.reset()
        assert self.cb.state() == STATE_CLOSED
        assert self.cb.allow_request()

    def test_persists_across_instances(self, tmp_path: Path):
        path = str(tmp_path / "cb.json")
        cb1 = CircuitBreaker("test", CircuitBreakerConfig(
            window_seconds=10, min_calls=2, failure_threshold=0.5, persist_path=path,
        ))
        for _ in range(5):
            cb1.record_failure()
        assert cb1.state() == STATE_OPEN

        cb2 = CircuitBreaker("test", CircuitBreakerConfig(
            window_seconds=10, min_calls=2, failure_threshold=0.5, persist_path=path,
        ))
        assert cb2.state() == STATE_OPEN


# ── AdaptiveBackoff ────────────────────────────────────────────────────────


class TestAdaptiveBackoff:
    def setup_method(self) -> None:
        self.b = AdaptiveBackoff(AdaptiveBackoffConfig(
            base_seconds=1, max_seconds=10, factor=2.0,
            static_intervals=((1, 5), (2, 10), (3, 20)),
        ))

    def test_initial_should_retry(self):
        assert self.b.should_retry("X")

    def test_404_uses_static_schedule(self):
        t1 = self.b.record_failure("ROG.SW", KIND_404)
        # First failure → 5s
        assert t1 - time.time() >= 4.5
        t2 = self.b.record_failure("ROG.SW", KIND_404)
        # Second failure → 10s
        assert t2 - time.time() >= 9.5
        t3 = self.b.record_failure("ROG.SW", KIND_404)
        # Third failure → 20s (capped)
        assert t3 - time.time() >= 19.5

    def test_5xx_uses_exponential(self):
        t1 = self.b.record_failure("X", KIND_5XX)
        # First failure → base 1s
        assert 0.5 <= t1 - time.time() <= 1.5
        t2 = self.b.record_failure("X", KIND_5XX)
        # Second → 2s
        assert 1.5 <= t2 - time.time() <= 2.5
        t3 = self.b.record_failure("X", KIND_5XX)
        # Third → 4s
        assert 3.5 <= t3 - time.time() <= 4.5

    def test_caps_at_max(self):
        for _ in range(10):
            self.b.record_failure("X", KIND_5XX)
        # Should be capped at 10s
        assert self.b.next_retry_at("X") - time.time() <= 10.5

    def test_should_retry_respects_window(self):
        self.b.record_failure("X", KIND_5XX)
        assert not self.b.should_retry("X")
        time.sleep(1.1)
        assert self.b.should_retry("X")

    def test_success_clears(self):
        self.b.record_failure("X", KIND_5XX)
        assert not self.b.should_retry("X")
        self.b.record_success("X")
        assert self.b.should_retry("X")

    def test_status(self):
        assert self.b.status("X")["failures"] == 0
        self.b.record_failure("X", KIND_5XX)
        s = self.b.status("X")
        assert s["failures"] == 1
        assert s["last_kind"] == KIND_5XX
        assert s["next_retry_at"] is not None


# ── ResilienceGuard (the integration) ──────────────────────────────────────


class TestResilienceGuard:
    def setup_method(self) -> None:
        # Use a relaxed breaker so the per-key behaviours can be tested
        # independently. The breaker-only test below uses .breaker directly.
        from app.data.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from app.data.negative_cache import NegativeCache, NegativeCacheConfig
        from app.data.adaptive_backoff import AdaptiveBackoff, AdaptiveBackoffConfig
        relaxed_breaker = CircuitBreaker(
            "test-guard",
            CircuitBreakerConfig(
                window_seconds=10, min_calls=100, failure_threshold=0.9,
                reset_timeout_seconds=1,
            ),
        )
        # Negative cache needs a *short* skip so the success-reset test
        # can verify the gate flips back without sleeping.
        fast_negative = NegativeCache(
            name="test-guard",
            config=NegativeCacheConfig(
                failure_threshold=2, window_seconds=60,
                skip_duration=60, max_skip_duration=120,
            ),
        )
        fast_backoff = AdaptiveBackoff(
            name="test-guard",
            config=AdaptiveBackoffConfig(base_seconds=60, max_seconds=120, factor=2.0),
        )
        self.guard = ResilienceGuard(
            "test-guard",
            breaker=relaxed_breaker,
            negative=fast_negative,
            backoff=fast_backoff,
        )

    def test_normal_call_proceeds(self):
        d = self.guard.before_call("KEY")
        assert d.proceed
        assert d.reason == "ok"

    def test_breaker_open_short_circuits(self):
        # Use a separate guard with a tight breaker
        from app.data.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        tight = CircuitBreaker("tight", CircuitBreakerConfig(
            window_seconds=10, min_calls=3, failure_threshold=0.5, reset_timeout_seconds=60,
        ))
        guard = ResilienceGuard("tight", breaker=tight)
        for _ in range(20):
            guard.breaker.record_failure()
        d = guard.before_call("KEY")
        assert not d.proceed
        assert d.reason == "breaker open"
        assert d.state == STATE_OPEN

    def test_backoff_blocks_call(self):
        self.guard.after_failure("KEY", KIND_5XX)
        d = self.guard.before_call("KEY")
        assert not d.proceed
        assert "backoff" in d.reason

    def test_negative_cache_skips(self):
        # Force the negative cache to SKIP without involving the breaker
        for _ in range(3):
            self.guard.negative.record("KEY", NC_5XX)
        d = self.guard.before_call("KEY")
        # Backoff will block first (we just recorded 3 failures). Reset
        # the backoff to isolate the negative-cache behaviour.
        self.guard.backoff.reset("KEY")
        d = self.guard.before_call("KEY")
        assert not d.proceed
        assert "negative-cache" in d.reason
        assert d.state == "skip"

    def test_success_resets_all(self):
        for _ in range(3):
            self.guard.after_failure("KEY", KIND_5XX)
        assert not self.guard.before_call("KEY").proceed
        self.guard.after_success("KEY")
        # Breaker was never tripped (min_calls=100), so we can call again
        assert self.guard.before_call("KEY").proceed

    def test_stats(self):
        self.guard.after_failure("A", KIND_5XX)
        self.guard.after_failure("B", KIND_404)
        s = self.guard.stats()
        assert s["name"] == "test-guard"
        assert "breaker" in s
        assert "negative_cache" in s
        assert "backoff" in s


class TestResilientContextManager:
    def test_ok_path(self):
        with resilient("test", "KEY") as r:
            assert r.should_call
            r.ok()
        # No failure recorded → guard is healthy

    def test_exception_records_failure(self):
        with pytest.raises(RuntimeError):
            with resilient("test", "KEY") as r:
                assert r.should_call
                raise RuntimeError("bridge down")
        # After the exception, the failure was recorded
        g = for_source("test")
        assert g.backoff.status("KEY")["failures"] == 1

    def test_skip_path(self):
        g = for_source("test")
        for _ in range(20):
            g.after_failure("KEY2", KIND_5XX)
        with resilient("test", "KEY2") as r:
            assert not r.should_call


class TestSourcePresets:
    def test_known_sources_have_presets(self):
        mt5 = for_source("mt5")
        yahoo = for_source("yahoo")
        assert mt5.name == "mt5"
        assert yahoo.name == "yahoo"
        # Preset-specific tuning: MT5 has tighter breaker (30s vs 60s)
        assert mt5.breaker.config.window_seconds == 30
        assert yahoo.breaker.config.window_seconds == 60
        # Yahoo: 404s (permanent) get a longer static schedule
        assert yahoo.backoff.config.static_intervals[0][1] == 300  # 5 min

    def test_unknown_source_uses_defaults(self):
        g = for_source("unknown-source-xyz")
        assert g.breaker.config.failure_threshold == 0.5  # default


class TestClassifyException:
    def test_404(self):
        assert classify_exception(RuntimeError("HTTP 404 not found")) == KIND_404
    def test_429(self):
        assert classify_exception(RuntimeError("429 rate limit exceeded")) == KIND_429
    def test_timeout(self):
        assert classify_exception(RuntimeError("connection timed out")) == KIND_TIMEOUT
    def test_5xx(self):
        assert classify_exception(RuntimeError("HTTP 502 bad gateway")) == KIND_5XX
        assert classify_exception(RuntimeError("service unavailable")) == KIND_5XX
    def test_other(self):
        assert classify_exception(RuntimeError("weird error")) == KIND_OTHER
