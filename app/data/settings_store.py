"""Persistent key-value settings store for provider API keys.

The production app uses PostgreSQL (via Prisma) for this. The Python
reference uses a JSON file at `data/settings.json` — same semantics, just
a different backend.

Why
---
API keys (OpenRouter, Minimax, Finnhub, Twelve Data, Alpha Vantage, …)
need to be:
  * persistent across container restarts (so docker-compose rebuilds
    don't lose them)
  * modifiable at runtime through the WebUI (no redeploy to change a key)
  * overridable per-request via header (useful for testing, BYO-key)
  * not committed to source control

This module provides a single, thread-safe read/write API. The store is
loaded once and cached in memory; writes flush to disk atomically (write
to a temp file, then rename).

Resolution order (in `get()`)
-----------------------------
1. Per-request override (passed by the caller, e.g. from a request header)
2. The persistent store (DB in production, JSON in Python reference)
3. Environment variable (operator-configured fallback)
4. Empty string (provider is unavailable)

Node.js porting notes
---------------------
The same 4-level resolution applies. In Node:
  1. const requestKey = req.headers["x-provider-key-finnhub"];
  2. const dbKey = await prisma.setting.findUnique({ where: { key: "FINNHUB_API_KEY" } });
  3. const envKey = process.env.FINNHUB_API_KEY;
  4. ""
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Well-known keys (used as the canonical name in the store and as env var name)
KEY_FINNHUB = "FINNHUB_API_KEY"
KEY_TWELVE_DATA = "TWELVE_DATA_API_KEY"
KEY_ALPHA_VANTAGE = "ALPHA_VANTAGE_API_KEY"
KEY_OPENROUTER = "OPENROUTER_API_KEY"
KEY_MINIMAX = "MINIMAX_API_KEY"
KEY_FINNHUB_TOKEN = "FINNHUB_TOKEN"  # alternative name

# All known provider keys (provider_name → canonical key name)
PROVIDER_KEYS = {
    "finnhub": KEY_FINNHUB,
    "twelve_data": KEY_TWELVE_DATA,
    "alpha_vantage": KEY_ALPHA_VANTAGE,
    "openrouter": KEY_OPENROUTER,
    "minimax": KEY_MINIMAX,
}


class SettingsStore:
    """Thread-safe, file-backed settings store.

    Reads from a JSON file (`data/settings.json` by default). On first read
    the file is loaded into memory; subsequent reads are O(1) from RAM.
    Writes flush atomically to disk.

    In production, replace the file backend with a Prisma / SQLAlchemy
    client — the public API stays the same.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "data" / "settings.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._load()

    def get(self, key: str, default: str = "") -> str:
        """Read a value. Returns `default` (empty string) if not set."""
        with self._lock:
            return self._cache.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Write a value. Empty string removes the key."""
        with self._lock:
            if value:
                self._cache[key] = value
            else:
                self._cache.pop(key, None)
            self._flush()
        logger.debug("SettingsStore: %s = %s", key, "***" if value else "(cleared)")

    def delete(self, key: str) -> None:
        self.set(key, "")

    def all(self) -> dict[str, str]:
        """Return all settings (for /api/config, debug endpoints, etc.)."""
        with self._lock:
            return dict(self._cache)

    def has(self, key: str) -> bool:
        return bool(self.get(key))

    # ── Provider-specific helpers ───────────────────────────────────────────

    def get_provider_key(
        self,
        provider: str,
        request_key: str = "",
    ) -> str:
        """Resolve the API key for a provider using the 4-level priority.

        Args:
            provider: "finnhub" / "twelve_data" / "alpha_vantage" / "minimax" / "openrouter"
            request_key: optional per-request override (e.g. from HTTP header)

        Returns the first non-empty value:
            1. request_key
            2. store (data/settings.json / Postgres)
            3. environment variable
            4. ""
        """
        # 1. Per-request override
        if request_key and request_key.strip():
            return request_key.strip()

        # 2. Persistent store
        canonical = PROVIDER_KEYS.get(provider.lower(), f"{provider.upper()}_API_KEY")
        v = self.get(canonical)
        if v:
            return v.strip()

        # 3. Environment fallback
        env_v = os.environ.get(canonical, "").strip()
        if env_v:
            return env_v

        # 4. Empty
        return ""

    def is_provider_configured(self, provider: str, request_key: str = "") -> bool:
        return bool(self.get_provider_key(provider, request_key))

    # ── Internals ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._cache = {k: str(v) for k, v in data.items() if v}
        except Exception as exc:
            logger.warning("SettingsStore: failed to load %s: %s", self._path, exc)
            self._cache = {}

    def _flush(self) -> None:
        """Atomic write: tmp file + rename, so concurrent readers never see a half-written file."""
        try:
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("SettingsStore: failed to flush %s: %s", self._path, exc)


# ── Module-level singleton ─────────────────────────────────────────────────

_default: SettingsStore | None = None
_default_lock = threading.Lock()


def get_store() -> SettingsStore:
    """Return the process-wide SettingsStore (lazy-initialized)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = SettingsStore()
        return _default
