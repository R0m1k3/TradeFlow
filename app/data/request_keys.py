"""Per-request API-key resolution for the multi-source data layer.

Implements the BYO-key pattern: the API key is sent by the frontend in
the `X-Provider-Key-<NAME>` header, and the backend uses it for the
duration of one HTTP request only. The key is never persisted server-side.

Why
---
Many self-hosted apps let users plug in their own API keys (OpenAI, Finnhub,
Twelve Data, etc.) instead of asking the operator to set up server-side
credentials. This module makes that pattern work cleanly with the
existing SourceRouter / provider stack.

Usage
-----
Frontend sends:
    X-Provider-Key-Finnhub: d7lu8...
    X-Provider-Key-TwelveData: xxx
    X-Provider-Key-AlphaVantage: yyy

Backend does:
    from app.data.request_keys import get_provider_key

    finnhub = FinnhubProvider()
    finnhub.set_request_key(get_provider_key(request, "finnhub"))
    df = finnhub.fetch_ohlcv("AAPL")

    # or via the router:
    router = SourceRouter.default()
    result = router.fetch_ohlcv("AAPL", keys=get_keys_from_request(request))

Headers
-------
* `X-Provider-Key-Finnhub`
* `X-Provider-Key-TwelveData`
* `X-Provider-Key-AlphaVantage`

Casing doesn't matter; the lookup is case-insensitive.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Header name pattern
HEADER_PREFIX = "X-Provider-Key-"

# Map provider name → env-var name (used as fallback when header is absent)
PROVIDER_ENV_FALLBACK = {
    "finnhub": "FINNHUB_API_KEY",
    "twelve_data": "TWELVE_DATA_API_KEY",
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
}

# Provider name normalization for header lookup
PROVIDER_KEY_NORMALIZE = {
    "finnhub": "Finnhub",
    "twelve_data": "TwelveData",
    "twelve-data": "TwelveData",
    "alpha_vantage": "AlphaVantage",
    "alpha-vantage": "AlphaVantage",
}


def get_provider_key(headers: dict | None, provider: str) -> str:
    """Return the API key for a given provider from request headers.

    Resolution order:
        1. `X-Provider-Key-<Name>` header (BYO-key, per-request)
        2. `Authorization: Bearer <key>` (OpenAI-style)
        3. Server-side env var (operator-configured)
        4. Empty string

    Never raises. The provider itself decides what to do when the key
    is empty (typically: skip the call and let the router try the
    next source).
    """
    headers = headers or {}
    provider = (provider or "").lower().replace("-", "_")

    # 1. Custom header
    name = PROVIDER_KEY_NORMALIZE.get(provider, provider.replace("_", "").title())
    header_name = HEADER_PREFIX + name
    for k, v in headers.items():
        if k.lower() == header_name.lower() and v:
            return v.strip()

    # 2. Authorization: Bearer <key>
    auth = None
    for k, v in headers.items():
        if k.lower() == "authorization" and v:
            auth = v.strip()
            break
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()

    # 3. Server-side env fallback
    import os
    env_name = PROVIDER_ENV_FALLBACK.get(provider, "")
    if env_name:
        return os.environ.get(env_name, "").strip()

    return ""


def get_all_provider_keys(headers: dict | None) -> dict[str, str]:
    """Return all provider keys resolved from a request, keyed by provider name.

    Used by SourceRouter to inject per-request keys into providers before
    each call. Returns a dict like:
        {"finnhub": "d7...", "twelve_data": "xxx", "alpha_vantage": ""}
    """
    headers = headers or {}
    out = {}
    for provider in PROVIDER_ENV_FALLBACK:
        out[provider] = get_provider_key(headers, provider)
    return out
