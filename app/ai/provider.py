"""Unified AI provider abstraction — Minimax (Anthropic-compatible) + OpenRouter.

Goals:
  * Single entry point (`call_ai`) so the scheduler / webui / bot share one path.
  * Built-in retries with exponential backoff + jitter.
  * Persistent cache (`app.ai.cache`) keyed on (provider, model, prompt, mode).
  * Async + sync helpers; sync falls back to `httpx` if asyncio isn't available.
  * Cheap text extraction: many LLM responses wrap JSON in markdown fences —
    we strip those before parsing.

Providers:
  - "minimax"   → https://api.minimax.io/anthropic  (Anthropic Messages API)
                  Default model: "MiniMax-M3"
                  Auth header: x-api-key
  - "openrouter"→ https://openrouter.ai/api/v1/chat/completions
                  Default model: "perplexity/sonar"
                  Auth header: Authorization: Bearer ...

If `provider == "auto"`, the first provider with a configured key is used
(Minimax first → cheaper / in-house, then OpenRouter).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from app.ai import cache

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MINIMAX_ENDPOINT = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"
MINIMAX_DEFAULT_MAX_TOKENS = 1024
MINIMAX_VERSION = "2023-06-01"

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
OPENROUTER_DEFAULT_MODEL = "perplexity/sonar"

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "settings.json"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# ── Provider config dataclass ────────────────────────────────────────────────


@dataclass
class AIConfig:
    provider: str = "auto"        # "minimax" | "openrouter" | "auto"
    model: str = ""               # model id; "" → provider default
    api_key: str = ""             # explicit key (overrides env)
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_seconds: int = 1800
    minimax_model: str = MINIMAX_DEFAULT_MODEL
    openrouter_model: str = OPENROUTER_DEFAULT_MODEL
    # set to True to force JSON parsing of the response (raises on failure)
    require_json: bool = True


# ── Key resolution (settings → env → .env) ──────────────────────────────────


def _read_settings_key(name: str) -> str:
    try:
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            v = data.get(name, "")
            if v:
                return str(v)
    except Exception:
        pass
    return ""


def _read_env_file_key(name: str) -> str:
    try:
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{name}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    except Exception:
        pass
    return ""


def _resolve_key(env_names: tuple[str, ...]) -> str:
    """Look up the first non-empty value across: settings.json, os.environ, .env."""
    for n in env_names:
        v = _read_settings_key(n)
        if v:
            return v
    for n in env_names:
        v = os.environ.get(n, "")
        if v:
            return v
    for n in env_names:
        v = _read_env_file_key(n)
        if v:
            return v
    return ""


def get_minimax_key() -> str:
    return _resolve_key(("MINIMAX_API_KEY", "ANTHROPIC_AUTH_TOKEN"))


def get_openrouter_key() -> str:
    return _resolve_key(("OPENROUTER_API_KEY",))


# ── YAML config loading ─────────────────────────────────────────────────────


def _load_yaml() -> dict:
    try:
        import yaml
        if CONFIG_PATH.exists():
            return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.debug("YAML load failed: %s", exc)
    return {}


def load_ai_config(ai_cfg: dict | None = None) -> AIConfig:
    """Build an AIConfig from YAML + env. `ai_cfg` overrides the YAML block."""
    yaml_cfg = _load_yaml().get("ai_analysis", {}) or {}
    merged = {**yaml_cfg, **(ai_cfg or {})}

    # Determine provider + key + model, honouring explicit api_key first.
    provider = str(merged.get("provider", "auto")).lower()
    if provider not in ("minimax", "openrouter", "auto"):
        provider = "auto"

    model = str(merged.get("model", "") or "").strip()

    # Resolve keys lazily — at call time we re-check so UI updates take effect.
    cfg = AIConfig(
        provider=provider,
        model=model,
        timeout_seconds=int(merged.get("timeout_seconds", 30)),
        max_retries=int(merged.get("max_retries", 3)),
        cache_ttl_seconds=int(merged.get("cache_ttl_seconds", merged.get("score_ttl_seconds", 1800))),
        minimax_model=str(merged.get("minimax_model", MINIMAX_DEFAULT_MODEL)),
        openrouter_model=str(merged.get("openrouter_model", OPENROUTER_DEFAULT_MODEL)),
        require_json=bool(merged.get("require_json", True)),
    )
    return cfg


def resolve_active_provider(cfg: AIConfig) -> tuple[str, str, str]:
    """Return (provider, api_key, model) based on cfg.provider + available keys."""
    minimax_key = get_minimax_key()
    openrouter_key = get_openrouter_key()

    if cfg.provider == "minimax":
        return ("minimax", minimax_key, cfg.model or cfg.minimax_model)
    if cfg.provider == "openrouter":
        return ("openrouter", openrouter_key, cfg.model or cfg.openrouter_model)
    # auto: prefer minimax when available
    if minimax_key:
        return ("minimax", minimax_key, cfg.model or cfg.minimax_model)
    if openrouter_key:
        return ("openrouter", openrouter_key, cfg.model or cfg.openrouter_model)
    return ("", "", cfg.model)


# ── Response parsing helpers ────────────────────────────────────────────────


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_text_and_json(content: Any) -> tuple[str, Optional[Any]]:
    """Normalize the provider response to a text string and (if present) a parsed object.

    Accepts:
      * str (raw model output)
      * list of Anthropic-style blocks: [{"type": "text", "text": "..."}]
      * OpenAI-style dict: {"choices": [{"message": {"content": "..."}}]}
    Returns: (text, parsed_dict_or_None)
    """
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Anthropic content blocks
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
    elif isinstance(content, dict):
        # OpenAI-style already parsed
        try:
            text = content["choices"][0]["message"]["content"]
        except Exception:
            text = json.dumps(content, ensure_ascii=False)
    else:
        text = str(content)

    parsed: Optional[Any] = None
    if not text:
        return text, parsed

    # Try markdown fence first
    m = _JSON_FENCE.search(text)
    candidate = m.group(1) if m else text
    # Trim
    candidate = candidate.strip()

    # Direct JSON
    try:
        parsed = json.loads(candidate)
        return text, parsed
    except Exception:
        pass

    # First balanced JSON object in the text
    if "{" in candidate:
        start = candidate.find("{")
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start : i + 1]
                    try:
                        parsed = json.loads(snippet)
                        return text, parsed
                    except Exception:
                        break
    return text, None


# ── Provider call: Minimax (Anthropic-compatible) ───────────────────────────


def _minimax_chat(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
) -> Any:
    """Sync call to Minimax (Anthropic Messages API). Returns the raw parsed JSON."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": MINIMAX_VERSION,
        "content-type": "application/json",
    }
    # Split system message if present (Anthropic expects top-level `system`).
    system = None
    user_messages: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            user_messages.append(m)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_messages,
    }
    if system:
        payload["system"] = system
    with httpx.Client(timeout=timeout) as client:
        r = client.post(MINIMAX_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


# ── Provider call: OpenRouter (OpenAI-compatible) ───────────────────────────


def _openrouter_chat(
    messages: list[dict],
    model: str,
    api_key: str,
    timeout: int,
    response_format_json: bool = True,
) -> Any:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://tradeflow.local",
        "X-Title": "TradeFlow",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(OPENROUTER_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


# ── Retry wrapper ───────────────────────────────────────────────────────────


def _with_retries(fn, *, retries: int, base_delay: float = 0.6) -> Any:
    """Run `fn` with exponential backoff on transient errors (5xx, network, 429)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            transient = status == 429 or 500 <= status < 600
            if not transient or attempt == retries:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            logger.warning(
                "AI provider %s on attempt %d/%d (status %d) — retrying in %.2fs",
                exc.request.url.host, attempt + 1, retries + 1, status, sleep_s,
            )
            time.sleep(sleep_s)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt == retries:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            logger.warning("AI network error on attempt %d/%d: %s — retrying in %.2fs",
                           attempt + 1, retries + 1, exc, sleep_s)
            time.sleep(sleep_s)
    # Unreachable, but keep mypy happy
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


# ── Public API ──────────────────────────────────────────────────────────────


def call_ai(
    prompt: str,
    *,
    mode: str = "hybrid",
    cfg: AIConfig | None = None,
    system: str = "",
    max_tokens: int = MINIMAX_DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> dict:
    """Send a single prompt to the active provider, with caching and retries.

    Returns a normalized dict with at least:
        {"provider": ..., "model": ..., "text": ..., "json": parsed_or_None, "raw": ...}
    Raises on hard failure (no provider, no key, parse error when require_json).
    """
    cfg = cfg or load_ai_config()
    provider, api_key, model = resolve_active_provider(cfg)

    if not provider:
        raise RuntimeError("No AI provider configured (set MINIMAX_API_KEY or OPENROUTER_API_KEY)")
    if not api_key:
        raise RuntimeError(f"No API key for provider '{provider}'")
    if not model:
        raise RuntimeError(f"No model configured for provider '{provider}'")

    cache_ttl = cfg.cache_ttl_seconds if use_cache else 0
    if cache_ttl > 0:
        cached = cache.get(provider, model, prompt, ttl=cache_ttl, mode=mode)
        if cached is not None:
            cached = dict(cached)  # copy
            cached["cached"] = True
            return cached

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _do() -> dict:
        if provider == "minimax":
            raw = _minimax_chat(messages, model, api_key, max_tokens, cfg.timeout_seconds)
            content = raw.get("content", [])
        elif provider == "openrouter":
            raw = _openrouter_chat(messages, model, api_key, cfg.timeout_seconds)
            content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:  # defensive
            raise RuntimeError(f"Unknown provider: {provider}")
        text, parsed = _extract_text_and_json(content)
        return {"provider": provider, "model": model, "text": text, "json": parsed, "raw": raw}

    result = _with_retries(_do, retries=cfg.max_retries)

    if cfg.require_json and result.get("json") is None:
        raise RuntimeError(
            f"AI response from {provider}/{model} did not contain parseable JSON"
        )

    # Cache the slim payload (drop raw to keep disk size small)
    if cache_ttl > 0:
        slim = {
            "provider": result["provider"],
            "model": result["model"],
            "text": result["text"],
            "json": result["json"],
        }
        cache.set(provider, model, prompt, slim, mode=mode)
    return result


# ── Backwards-compat helpers (used by scheduler / webui) ───────────────────


def fetch_ai_score(ticker: str, model: str, api_key: str, timeout: int = 30) -> tuple[float, str, list]:
    """Legacy OpenRouter-style helper. Kept for compatibility with existing callers."""
    from app.ai.openrouter_client import PROMPT_TEMPLATE, fetch_ai_score as _or_score
    return _or_score(ticker, model, api_key, timeout)


def test_connection(provider: str, api_key: str, model: str | None = None, timeout: int = 15) -> dict:
    """Quick connectivity check. Returns {ok, provider, model, error?}.

    Works for both Minimax and OpenRouter; picks a sensible default model
    if `model` is empty.
    """
    if not api_key:
        return {"ok": False, "provider": provider, "error": "no api key"}

    test_prompt = (
        'Reply with valid JSON only: {"score": 0.5, "rationale": "test", "sources": []}'
    )
    system = ""
    cfg = AIConfig(
        provider=provider,
        model=model or (MINIMAX_DEFAULT_MODEL if provider == "minimax" else OPENROUTER_DEFAULT_MODEL),
        api_key=api_key,
        timeout_seconds=timeout,
        max_retries=1,
        cache_ttl_seconds=0,
    )
    try:
        result = call_ai(test_prompt, mode="test", cfg=cfg, system=system, use_cache=False)
        return {
            "ok": result.get("json") is not None,
            "provider": result["provider"],
            "model": result["model"],
        }
    except Exception as exc:
        return {"ok": False, "provider": provider, "model": cfg.model, "error": str(exc)}


def list_openrouter_models(api_key: str = "", timeout: int = 10) -> list[str]:
    """Return sorted list of model IDs available on OpenRouter (helper for the UI)."""
    headers = {"HTTP-Referer": "https://tradeflow.local", "X-Title": "TradeFlow"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(OPENROUTER_MODELS_ENDPOINT, headers=headers)
            r.raise_for_status()
        models = r.json().get("data", [])
        return sorted(m["id"] for m in models if "id" in m)
    except Exception as exc:
        logger.warning("Could not fetch OpenRouter model list: %s", exc)
        return []


# ── Provider status (for /api/ai/status) ───────────────────────────────────


def provider_status() -> dict:
    """Return a snapshot of what's configured and reachable."""
    minimax_key = get_minimax_key()
    openrouter_key = get_openrouter_key()
    return {
        "minimax": {
            "key_configured": bool(minimax_key),
            "endpoint": MINIMAX_ENDPOINT,
            "default_model": MINIMAX_DEFAULT_MODEL,
        },
        "openrouter": {
            "key_configured": bool(openrouter_key),
            "endpoint": OPENROUTER_ENDPOINT,
            "default_model": OPENROUTER_DEFAULT_MODEL,
        },
        "cache": cache.stats(),
    }
