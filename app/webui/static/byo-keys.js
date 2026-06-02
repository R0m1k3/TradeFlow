/**
 * byo-keys.js — BYO-Key helper for the multi-source data layer.
 *
 * Usage in the frontend (browser, React/Streamlit/vanilla):
 *
 *   <script src="/static/byo-keys.js"></script>
 *   <script>
 *     // 1. User types keys (in a settings modal):
 *     byoKeys.set({ finnhub: "d7lu8...", twelveData: "xxx", alphaVantage: "yyy" });
 *
 *     // 2. Every API call automatically includes the keys as headers:
 *     const r = await byoKeys.fetch("/api/admin/test-source?symbol=AAPL", { method: "POST" });
 *
 *     // 3. Inspect what's configured:
 *     byoKeys.get();   // { finnhub: "d7...", twelveData: "xxx", alphaVantage: "yyy" }
 *     byoKeys.hasAny(); // true if at least one key is set
 *
 * Storage
 * -------
 * Keys are stored in `localStorage` under the key `tradeflow:byo-keys`
 * as a JSON object. The user can clear them at any time with `byoKeys.clear()`.
 *
 * Security note
 * -------------
 * These keys live in the user's own browser. They are sent only to the
 * user's own backend (trade.ffnancy.fr) over HTTPS. They are NEVER sent
 * to any third-party directly from the browser.
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "tradeflow:byo-keys";

  // Map our friendly name → backend header suffix
  var NAME_TO_HEADER = {
    finnhub: "Finnhub",
    twelveData: "TwelveData",
    twelve_data: "TwelveData",
    alphaVantage: "AlphaVantage",
    alpha_vantage: "AlphaVantage",
  };

  function _load() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      var parsed = JSON.parse(raw);
      return (parsed && typeof parsed === "object") ? parsed : {};
    } catch (e) {
      return {};
    }
  }

  function _save(obj) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(obj || {}));
    } catch (e) {
      console.warn("[byoKeys] localStorage save failed:", e);
    }
  }

  function get() {
    return _load();
  }

  function set(partial) {
    var current = _load();
    var merged = Object.assign({}, current, partial || {});
    // Drop empty strings (treat as "clear this key")
    Object.keys(merged).forEach(function (k) {
      if (merged[k] === null || merged[k] === undefined || merged[k] === "") {
        delete merged[k];
      }
    });
    _save(merged);
    return merged;
  }

  function clear(name) {
    if (name) {
      var current = _load();
      delete current[name];
      _save(current);
      return current;
    }
    _save({});
    return {};
  }

  function hasAny() {
    var k = _load();
    return Object.keys(k).length > 0;
  }

  function _buildHeaders() {
    var keys = _load();
    var headers = {};
    Object.keys(keys).forEach(function (name) {
      var headerSuffix = NAME_TO_HEADER[name] || name;
      headers["X-Provider-Key-" + headerSuffix] = keys[name];
    });
    return headers;
  }

  /**
   * Drop-in replacement for fetch() that automatically attaches the
   * X-Provider-Key-* headers for every configured provider.
   */
  function fetch(url, options) {
    options = options || {};
    options.headers = Object.assign({}, _buildHeaders(), options.headers || {});
    return global.fetch(url, options);
  }

  global.byoKeys = {
    get: get,
    set: set,
    clear: clear,
    hasAny: hasAny,
    fetch: fetch,
    _buildHeaders: _buildHeaders,
    STORAGE_KEY: STORAGE_KEY,
  };
})(typeof window !== "undefined" ? window : globalThis);
