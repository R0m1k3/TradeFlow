# Multi-Source Data Layer

The app needs OHLCV + quote data for ~150 tickers across NASDAQ, LSE,
Euronext (Paris/Amsterdam), and SIX (Swiss). Single-source dependencies
have failed in production — MT5 bridge goes down for hours, Yahoo
rate-limits or 404s on Swiss tickers, Finnhub occasionally 502s.

This package solves the problem with a **priority chain of providers**
plus a **resilience layer** (negative cache + circuit breaker + adaptive
backoff) that prevents any one source from dragging the system down.

## Architecture

```
fetch_ohlcv("ROG.SW", "1d", "3mo")
            │
            ▼
    SourceRouter.fetch_ohlcv()
            │
            ├─ Finnhub        (priority 1)  ─→ ProviderError 5xx ─→ next
            ├─ Twelve Data    (priority 2)  ─→ ProviderError 404 ─→ next
            ├─ Alpha Vantage  (priority 3)  ─→ ProviderError 429 ─→ next
            └─ Yahoo          (priority 4)  ─→ ProviderError 404 ─→ no data
            │
            ▼
    First success returned, with the source tagged.
```

Every call goes through the resilience layer:

```
guard.before_call("ROG.SW:1d:3mo")
    │
    ├─ circuit breaker open ?     → "breaker open"      → serve stale
    ├─ backoff window not elapsed? → "backoff active"    → serve stale
    ├─ negative cache skip?        → "negative-cache skip" → serve stale
    └─ all clear                   → make the real call
```

## Why these providers?

| Provider | Free tier | Coverage | Strengths | Weaknesses |
|---|---|---|---|---|
| **Finnhub** | 60 calls/min | US + EU + Asia + CH | Fast, reliable, WebSocket too | Requires key |
| **Twelve Data** | 800 calls/day, 8/min | US + EU + UK + CH + DE + FR + NL | Best European coverage including `.SW`, `.DE`, `.PA` | Requires key |
| **Alpha Vantage** | 25 calls/day, 5/min | US + EU + Asia | Free, simple API | Strict rate limits |
| **Yahoo (yfinance)** | unlimited | US + EU + most | No key required | 404s on weird tickers, rate-limited |

For the European markets that were broken in production:
- **LSE (.L)** — Finnhub, Twelve Data, Alpha Vantage all cover it
- **Euronext Paris (.PA)** — all three
- **Euronext Amsterdam (.AS)** — all three
- **SIX Swiss (.SW)** — Finnhub and Twelve Data (best)
- **Xetra Frankfurt (.DE)** — Twelve Data and Alpha Vantage

For NASDAQ/US:
- All four providers work, Finnhub is the fastest

## Provider implementations

Each provider implements the same interface:

```python
class BaseProvider:
    name: str
    def is_available(self) -> bool: ...
    def coverage(self) -> dict: ...
    def fetch_ohlcv(self, symbol, interval, period) -> Optional[pd.DataFrame]: ...
    def fetch_quote(self, symbol) -> Optional[float]: ...
```

Files:
- `providers/base.py` — abstract base + `ProviderError` with `kind` for 404/5xx/429
- `providers/finnhub_provider.py` — REST API, 60 calls/min free
- `providers/twelve_data_provider.py` — REST API, 800/day free, best EU coverage
- `providers/alpha_vantage_provider.py` — REST API, 25/day free
- `providers/yahoo_provider.py` — yfinance wrapper, always available
- `source_router.py` — priority chain + resilience integration

## Usage

```python
from app.data.source_router import SourceRouter

router = SourceRouter.default()
result = router.fetch_ohlcv("ROG.SW", interval="1d", period="3mo")
if result.df is not None:
    print(f"Got {len(result.df)} bars from {result.source} in {result.duration_ms}ms")
    # → "Got 64 bars from twelve_data in 230ms"
else:
    print(f"All sources failed: {result.tried}")
```

## Telemetry

`SourceRouter` integrates with the existing resilience layer. Every
provider has its own guard (circuit breaker, negative cache, backoff)
keyed by source name. The stats endpoint:

```python
from app.data.resilience_hook import all_stats
for guard in all_stats():
    print(guard["name"], guard["breaker"]["state"], guard["negative_cache"]["skipping"])
```

## Node.js port

The Python implementation is the reference. The Node.js production
server can implement the same pattern with:

```js
const providers = [
  new FinnhubProvider(process.env.FINNHUB_API_KEY),
  new TwelveDataProvider(process.env.TWELVE_DATA_API_KEY),
  new AlphaVantageProvider(process.env.ALPHA_VANTAGE_API_KEY),
  new YahooProvider(),
];

async function fetchOHLCV(symbol, interval, period) {
  for (const provider of providers) {
    if (!provider.isAvailable()) continue;
    const guard = resilience.forSource(provider.name);
    const decision = guard.beforeCall(`${symbol}:${interval}:${period}`);
    if (!decision.proceed) continue;
    try {
      const df = await provider.fetchOHLCV(symbol, interval, period);
      guard.afterSuccess(`${symbol}:${interval}:${period}`);
      return { df, source: provider.name };
    } catch (err) {
      guard.afterFailure(`${symbol}:${interval}:${period}`, classifyException(err));
    }
  }
  return null;
}
```

## API key setup

Each provider needs a free account. Total setup time: ~10 minutes for all 3.

| Provider | URL | Free tier |
|---|---|---|
| Finnhub | https://finnhub.io/register | 60 calls/min |
| Twelve Data | https://twelvedata.com/pricing | 800 calls/day |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | 25 calls/day |

Put the keys in `data/settings.json` (auto-saved by the WebUI) or
in `docker-compose.yml` as env vars:

```yaml
environment:
  - FINNHUB_API_KEY=your-key-here
  - TWELVE_DATA_API_KEY=your-key-here
  - ALPHA_VANTAGE_API_KEY=your-key-here
```
