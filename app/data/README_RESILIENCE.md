# Data Resilience Layer

Three composable primitives that protect the app from flaky external data
sources. Designed to be **portable to Node.js** (the production server uses
a Node stack — these modules are reference Python implementations).

## Why

When the MT5 bridge starts returning 502 for an entire market, or Yahoo
returns 404 for a Swiss ticker like `ROG.SW`, we don't want every poll to
re-hit the dead endpoint. The system should:

1. **Stop hammering** dead endpoints (don't make 95 requests that all fail).
2. **Serve stale-but-correct** data with a marker that it's stale.
3. **Auto-recover** when the upstream comes back, with minimal delay.

## The 3 primitives

| Module | Purpose | Key behaviour |
|---|---|---|
| `negative_cache.py` | Per-key failure tracking | After N failures in W seconds, *skip* the call for K seconds |
| `circuit_breaker.py` | Per-source health | 3-state (closed/open/half-open) with rolling failure rate |
| `adaptive_backoff.py` | Per-key retry interval | Static schedule for 404, exponential for 5xx/timeout/429 |
| `resilience_hook.py` | Glue + presets | One-stop `for_source("mt5")` registry with tuned defaults |

## Decision flow on every call

```
guard.before_call("ROG.SW:/ohlcv")
    │
    ├─ breaker.allow_request() ? ── no ─→ "breaker open"      → serve stale
    │
    ├─ backoff.should_retry(key) ? ── no ─→ "backoff active"   → serve stale
    │
    ├─ negative_cache.should_skip(key) ? ── yes → "negative-cache skip" → serve stale
    │
    └─ all clear → "ok" → make the real call
```

After the call:
- success → all 3 layers reset
- failure → all 3 layers record; the kind (`404`, `5xx`, `429`, `timeout`)
  drives the per-layer decision

## Usage (Python)

```python
from app.data.resilience_hook import for_source

# Built-in presets for known sources (mt5, yahoo, finnhub)
guard = for_source("mt5")
decision = guard.before_call(f"{symbol}:/ohlcv")
if not decision.proceed:
    return serve_cached_stale(symbol)

try:
    data = mt5_call(symbol)
    guard.after_success(f"{symbol}:/ohlcv")
    return data
except Exception as exc:
    from app.data.resilience_hook import classify_exception
    kind = classify_exception(exc)  # "404" / "5xx" / "timeout" / "429" / "other"
    guard.after_failure(f"{symbol}:/ohlcv", kind=kind)
    return serve_cached_stale(symbol)
```

## Usage (context manager — even cleaner)

```python
from app.data.resilience_hook import resilient

with resilient("mt5", f"{symbol}:/ohlcv") as r:
    if not r.should_call:
        return serve_cached_stale(symbol)
    try:
        data = mt5_call(symbol)
    except Exception:
        r.fail()  # kind auto-detected from exception
        return serve_cached_stale(symbol)
    r.ok()
    return data
```

## Built-in presets

| Source | Breaker window | Min calls | Fail threshold | Reset cooldown | Negative skip | Backoff base | 404 backoff |
|---|---|---|---|---|---|---|---|
| `mt5` | 30s | 5 | 50% | 60s | 120s → 900s | 30s | n/a |
| `yahoo` | 60s | 10 | 60% | 120s | 300s → 1800s | 60s | 300s static |
| `finnhub` | 30s | 5 | 50% | 60s | 60s → 300s | 60s | n/a |

## Telemetry

`GET /api/admin/resilience` returns the full state of every registered
guard — for the ops dashboard.

```json
{
  "guards": [
    {
      "name": "mt5",
      "breaker": { "state": "open", "failure_rate": 0.83, ... },
      "negative_cache": { "skipping": 47, "degraded": 12, ... },
      "backoff": { "active": 58, "blocked": 35, ... }
    },
    ...
  ]
}
```

## Node.js port

Each module is ~150 lines. The state machine in each is the only part
that needs careful porting — everything else is dict/Map manipulation.

Suggested Node layout:

```js
// negative-cache.js
class NegativeCache { ... }          // ~100 lines
module.exports = { NegativeCache, KIND_404, KIND_5XX, ... };

// circuit-breaker.js
class CircuitBreaker { ... }          // ~120 lines
module.exports = { CircuitBreaker, STATE_CLOSED, ... };

// adaptive-backoff.js
class AdaptiveBackoff { ... }         // ~100 lines

// resilience-hook.js
const { NegativeCache } = require('./negative-cache');
const { CircuitBreaker } = require('./circuit-breaker');
const { AdaptiveBackoff } = require('./adaptive-backoff');

const registry = new Map();
function forSource(name, preset) {
    if (!registry.has(name)) {
        registry.set(name, new ResilienceGuard(name, preset));
    }
    return registry.get(name);
}
```

The public API (`before_call`, `after_success`, `after_failure`, `stats`,
`reset`) maps 1-to-1 between languages.
