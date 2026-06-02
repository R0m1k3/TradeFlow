# Node.js Patch — BYO-Key Data Layer

Drop-in for the production Node.js server. Adds Finnhub + Twelve Data +
Alpha Vantage providers behind a priority chain with the resilience layer
(negative cache + circuit breaker + adaptive backoff).

Supports the **BYO-key pattern**: keys are sent by the frontend in
`X-Provider-Key-*` headers, never persisted server-side.

## Files to add

```
src/data/
  providers/
    base.js                 # Abstract base + ProviderError
    finnhub.js              # Finnhub REST (60 calls/min, free)
    twelve_data.js          # Twelve Data REST (800/day, best EU coverage)
    alpha_vantage.js        # Alpha Vantage REST (25/day, US+EU)
    yahoo.js                # yahoo-finance2 wrapper (always-on fallback)
  source_router.js          # Priority chain with resilience integration
  request_keys.js           # Header → provider key resolution
  resilience/
    negative_cache.js       # Per-key failure tracking
    circuit_breaker.js      # 3-state breaker
    adaptive_backoff.js     # Per-key retry interval
    index.js                # Combined ResilienceGuard + presets
```

## Files to modify

```
src/server.js               # Add /api/admin/test-source + /api/admin/providers
                            # Read X-Provider-Key-* headers
src/data/mt5_bridge.js      # Wire the existing MT5 client as a provider
public/static/byo-keys.js   # Already in the Python repo — copy as-is
```

## Quick install (assuming Node 20+, npm)

```bash
cd /opt/tradeflow
npm install axios          # if not already present
mkdir -p src/data/providers public/static

# Copy each file from this directory
cp app/data/NODEJS_PATCH/providers/*.js src/data/providers/
cp app/data/NODEJS_PATCH/resilience/*.js src/data/resilience/
cp src/data/source_router.js src/data/source_router.js
cp src/data/request_keys.js src/data/request_keys.js
cp public/static/byo-keys.js public/static/byo-keys.js
```

## Frontend wiring (one-time)

In your HTML page (the React SPA), include the helper and wrap your
fetch calls:

```html
<script src="/static/byo-keys.js"></script>
<script>
  // Settings modal: user enters keys
  byoKeys.set({
    finnhub: "d7lu8...",
    twelveData: "xxx",
    alphaVantage: "yyy"
  });

  // Every fetch automatically attaches X-Provider-Key-* headers
  const res = await byoKeys.fetch("/api/admin/test-source?symbol=AAPL", { method: "POST" });
</script>
```

## Server-side: 2 new endpoints

Add to your Express/Fastify router:

```js
// POST /api/admin/test-source
app.post("/api/admin/test-source", async (req, res) => {
  const { symbol, interval = "1d", period = "3mo" } = req.query;
  const keys = resolveProviderKeys(req.headers);  // request_keys.js

  const router = buildRouter(keys);  // see snippet below
  const result = await router.fetchOhlcv(symbol, interval, period);
  res.json(result);
});

// GET /api/admin/providers
app.get("/api/admin/providers", async (req, res) => {
  const keys = resolveProviderKeys(req.headers);
  res.json({
    providers: [
      new FinnhubProvider(keys.finnhub).stats(),
      new TwelveDataProvider(keys.twelveData).stats(),
      new AlphaVantageProvider(keys.alphaVantage).stats(),
      new YahooProvider().stats(),
    ],
    default_priority: ["finnhub", "twelve_data", "alpha_vantage", "yahoo"],
  });
});
```

## Router factory (drop-in)

```js
// src/data/source_router.js
import { FinnhubProvider } from "./providers/finnhub.js";
import { TwelveDataProvider } from "./providers/twelve_data.js";
import { AlphaVantageProvider } from "./providers/alpha_vantage.js";
import { YahooProvider } from "./providers/yahoo.js";
import { resilienceFor } from "./resilience/index.js";

export function buildRouter(keys = {}) {
  const providers = {
    finnhub: new FinnhubProvider(keys.finnhub),
    twelve_data: new TwelveDataProvider(keys.twelveData),
    alpha_vantage: new AlphaVantageProvider(keys.alphaVantage),
    yahoo: new YahooProvider(),
  };
  const priority = ["finnhub", "twelve_data", "alpha_vantage", "yahoo"];

  async function fetchOhlcv(symbol, interval = "1d", period = "3mo") {
    const tried = [];
    for (const name of priority) {
      const p = providers[name];
      if (!p.isAvailable()) continue;
      const guard = resilienceFor(name);
      const key = `${symbol}:${interval}:${period}`;
      const decision = guard.beforeCall(key);
      if (!decision.proceed) {
        tried.push(name);
        continue;
      }
      tried.push(name);
      try {
        const df = await p.fetchOhlcv(symbol, interval, period);
        guard.afterSuccess(key);
        return { df, source: name, tried };
      } catch (err) {
        guard.afterFailure(key, classifyException(err));
      }
    }
    return { df: null, source: "", tried };
  }

  return { fetchOhlcv, providers };
}
```

## Provider snippet (Finnhub)

```js
// src/data/providers/finnhub.js
import axios from "axios";

const ENDPOINT = "https://finnhub.io/api/v1";

const INTERVAL_MAP = {
  "1m": 1, "5m": 5, "15m": 15, "30m": 30,
  "1h": 60, "1d": "D", "1w": "W", "1M": "M",
};

export class FinnhubProvider {
  constructor(staticKey = "") {
    this.name = "finnhub";
    this._staticKey = staticKey;
    this._requestKey = "";
  }
  setRequestKey(k) { this._requestKey = (k || "").trim(); }
  clearRequestKey() { this._requestKey = ""; }
  _key() { return this._requestKey || this._staticKey; }
  isAvailable() { return !!this._key(); }
  coverage() { return { markets: ["US","EU","UK","CH","AS"], intervals: ["1m","5m","15m","30m","1h","1d","1w"], intraday: true, has_fundamentals: true }; }

  _resolve(symbol) { return symbol; }  // Yahoo symbols pass through

  async fetchOhlcv(symbol, interval = "1d", period = "3mo") {
    if (!this.isAvailable()) return null;
    if (!INTERVAL_MAP[interval]) throw new Error(`Unsupported interval: ${interval}`);
    const now = Math.floor(Date.now() / 1000);
    const from = now - periodToSeconds(period);
    try {
      const r = await axios.get(`${ENDPOINT}/stock/candle`, {
        params: { symbol: this._resolve(symbol), resolution: INTERVAL_MAP[interval], from, to: now },
        headers: { "X-Finnhub-Token": this._key() },
        timeout: 20000,
      });
      if (r.data.s !== "ok") throw new ProviderError("no_data", "404");
      const { t, o, h, l, c, v } = r.data;
      if (!t || !t.length) return null;
      return toDataFrame({ t, o, h, l, c, v });
    } catch (e) {
      throw mapAxiosError(e);
    }
  }

  async fetchQuote(symbol) {
    if (!this.isAvailable()) return null;
    try {
      const r = await axios.get(`${ENDPOINT}/quote`, {
        params: { symbol: this._resolve(symbol) },
        headers: { "X-Finnhub-Token": this._key() },
        timeout: 10000,
      });
      const price = r.data.c;
      if (!price) throw new ProviderError("no quote", "404");
      return price;
    } catch (e) {
      throw mapAxiosError(e);
    }
  }
}

class ProviderError extends Error {
  constructor(message, kind) { super(message); this.kind = kind; }
}

function mapAxiosError(e) {
  if (e.response?.status === 429) return new ProviderError("rate limit", "429");
  if (e.response?.status === 404) return new ProviderError("404", "404");
  if (e.response?.status >= 500) return new ProviderError(`${e.response.status}`, "5xx");
  if (e.code === "ECONNABORTED" || e.code === "ETIMEDOUT") return new ProviderError("timeout", "timeout");
  return new ProviderError(e.message, "other");
}

function periodToSeconds(p) {
  const m = p.match(/^(\d+)([dwmoy])$/i);
  if (!m) return 90 * 86400;
  const v = +m[1], u = m[2].toLowerCase();
  if (u === "d") return v * 86400;
  if (u === "w") return v * 7 * 86400;
  if (u === "mo" || u === "m") return v * 30 * 86400;
  if (u === "y") return v * 365 * 86400;
  return 90 * 86400;
}

function toDataFrame({ t, o, h, l, c, v }) {
  // Minimal: return a plain object with the same fields your OHLCV consumer expects.
  // Replace with whatever your codebase uses (e.g. danfo.js, simple-statistics).
  return {
    index: t.map(x => new Date(x * 1000)),
    open: o, high: h, low: l, close: c, volume: v,
  };
}
```

## Frontend BYO-key helper (already in Python repo at `app/webui/static/byo-keys.js`)

```js
// public/static/byo-keys.js
(function (g) {
  var KEY = "tradeflow:byo-keys";
  var N2H = { finnhub: "Finnhub", twelveData: "TwelveData", twelve_data: "TwelveData", alphaVantage: "AlphaVantage", alpha_vantage: "AlphaVantage" };
  function load() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { return {}; } }
  function save(o) { try { localStorage.setItem(KEY, JSON.stringify(o || {})); } catch (e) {} }
  function headers() {
    var k = load(), h = {};
    Object.keys(k).forEach(function (n) { h["X-Provider-Key-" + (N2H[n] || n)] = k[n]; });
    return h;
  }
  g.byoKeys = {
    get: load, set: function (p) { var c = Object.assign(load(), p || {}); Object.keys(c).forEach(function (k) { if (!c[k]) delete c[k]; }); save(c); return c; },
    clear: function (n) { if (n) { var c = load(); delete c[n]; save(c); return c; } save({}); return {}; },
    hasAny: function () { return Object.keys(load()).length > 0; },
    fetch: function (u, o) { o = o || {}; o.headers = Object.assign({}, headers(), o.headers || {}); return g.fetch(u, o); },
  };
})(typeof window !== "undefined" ? window : globalThis);
```

## Migration checklist

1. **Add files** (copies above) to your Node.js repo
2. **Mount the helper** in your HTML: `<script src="/static/byo-keys.js"></script>`
3. **Add the 2 endpoints** to your Express/Fastify router
4. **Replace your existing MT5-direct fetcher** with `buildRouter(keys).fetchOhlcv(...)`
5. **Add a settings UI** in the React SPA: 3 password inputs → `byoKeys.set({...})`
6. **Test**: open DevTools → Network → call `/api/admin/test-source?symbol=AAPL` → see `X-Provider-Key-Finnhub` header

That's it. The data layer is now multi-source with BYO-key support.
