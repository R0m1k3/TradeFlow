"""Unit tests for the unified AI provider and cache.

These tests don't hit the network — they exercise:
  * cache hit/miss/expiry
  * JSON fence / balanced-brace extraction
  * key resolution order
  * provider config parsing
  * `load_ai_config` and `resolve_active_provider`
  * retry/backoff (using a fake transport)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
import pytest

from app.ai import cache, provider
from app.ai.provider import (
    AIConfig,
    MINIMAX_DEFAULT_MODEL,
    OPENROUTER_DEFAULT_MODEL,
    _extract_text_and_json,
    load_ai_config,
    resolve_active_provider,
)


# ── Cache tests ──────────────────────────────────────────────────────────────


class TestCache:
    def setup_method(self) -> None:
        cache.clear()

    def test_set_and_get(self):
        cache.set("minimax", "MiniMax-M3", "hello", {"score": 0.7})
        got = cache.get("minimax", "MiniMax-M3", "hello", ttl=60)
        assert got == {"score": 0.7}

    def test_ttl_expiry(self):
        cache.set("minimax", "MiniMax-M3", "stale", {"score": 0.5})
        # 0-second TTL → always expired
        assert cache.get("minimax", "MiniMax-M3", "stale", ttl=0) is None

    def test_different_prompts_different_keys(self):
        cache.set("minimax", "MiniMax-M3", "prompt A", {"score": 0.1})
        cache.set("minimax", "MiniMax-M3", "prompt B", {"score": 0.9})
        assert cache.get("minimax", "MiniMax-M3", "prompt A", ttl=60) == {"score": 0.1}
        assert cache.get("minimax", "MiniMax-M3", "prompt B", ttl=60) == {"score": 0.9}

    def test_different_providers_different_keys(self):
        cache.set("minimax", "MiniMax-M3", "p", {"score": 0.1})
        cache.set("openrouter", "perplexity/sonar", "p", {"score": 0.9})
        assert cache.get("minimax", "MiniMax-M3", "p", ttl=60) == {"score": 0.1}
        assert cache.get("openrouter", "perplexity/sonar", "p", ttl=60) == {"score": 0.9}

    def test_mode_is_part_of_key(self):
        cache.set("minimax", "MiniMax-M3", "p", {"v": "hybrid"}, mode="hybrid")
        cache.set("minimax", "MiniMax-M3", "p", {"v": "auto"}, mode="autonomous")
        assert cache.get("minimax", "MiniMax-M3", "p", ttl=60, mode="hybrid") == {"v": "hybrid"}
        assert cache.get("minimax", "MiniMax-M3", "p", ttl=60, mode="autonomous") == {"v": "auto"}

    def test_persists_across_instances(self):
        """A fresh module load should still read the disk-backed cache."""
        cache.set("minimax", "MiniMax-M3", "persist", {"x": 1})
        # Simulate restart by clearing in-memory state
        cache._in_mem.clear()
        assert cache.get("minimax", "MiniMax-M3", "persist", ttl=60) == {"x": 1}

    def test_stats(self):
        cache.set("minimax", "MiniMax-M3", "p", {"a": 1})
        s = cache.stats()
        assert s["in_memory"] >= 1
        assert s["on_disk"] >= 1


# ── JSON extraction ─────────────────────────────────────────────────────────


class TestJsonExtraction:
    def test_plain_json(self):
        text, parsed = _extract_text_and_json('{"score": 0.5}')
        assert text == '{"score": 0.5}'
        assert parsed == {"score": 0.5}

    def test_markdown_fenced(self):
        text, parsed = _extract_text_and_json(
            "Here you go:\n```json\n{\"score\": 0.7}\n```\nBye."
        )
        assert parsed == {"score": 0.7}
        assert "Here you go" in text

    def test_anthropic_blocks(self):
        content = [
            {"type": "text", "text": "Sure!"},
            {"type": "text", "text": '{"score": 0.3, "rationale": "ok"}'},
        ]
        text, parsed = _extract_text_and_json(content)
        assert "Sure!" in text
        assert parsed == {"score": 0.3, "rationale": "ok"}

    def test_openai_dict(self):
        content = {"choices": [{"message": {"content": '{"score": 0.9}'}}]}
        text, parsed = _extract_text_and_json(content)
        assert parsed == {"score": 0.9}

    def test_unparseable_returns_none(self):
        text, parsed = _extract_text_and_json("just a string, no json here")
        assert text == "just a string, no json here"
        assert parsed is None

    def test_balanced_brace_fallback(self):
        # Leading text + nested JSON
        text, parsed = _extract_text_and_json(
            'Reasoning: something something. {"a": {"b": 1}} and that is all.'
        )
        assert parsed == {"a": {"b": 1}}


# ── Key resolution & config loading ─────────────────────────────────────────


class TestConfigLoading:
    def test_defaults(self):
        cfg = AIConfig()
        assert cfg.provider == "auto"
        assert cfg.timeout_seconds == 30
        assert cfg.max_retries == 3

    def test_load_from_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "ai_analysis:\n"
            "  enabled: true\n"
            "  provider: minimax\n"
            "  timeout_seconds: 45\n"
            "  minimax_model: MiniMax-M3\n"
        )
        # Patch the config path used by the module
        import yaml
        monkeypatch.setattr(provider, "CONFIG_PATH", cfg_file)
        cfg = load_ai_config()
        assert cfg.provider == "minimax"
        assert cfg.timeout_seconds == 45
        assert cfg.minimax_model == "MiniMax-M3"

    def test_resolve_active_provider_auto_prefers_minimax(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "sk-test")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "sk-or-test")
        cfg = AIConfig(provider="auto")
        prov, key, model = resolve_active_provider(cfg)
        assert prov == "minimax"
        assert key == "sk-test"
        assert model == MINIMAX_DEFAULT_MODEL

    def test_resolve_active_provider_falls_back_to_openrouter(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "sk-or")
        cfg = AIConfig(provider="auto")
        prov, key, model = resolve_active_provider(cfg)
        assert prov == "openrouter"
        assert key == "sk-or"
        assert model == OPENROUTER_DEFAULT_MODEL

    def test_resolve_no_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "")
        cfg = AIConfig(provider="auto")
        prov, key, _ = resolve_active_provider(cfg)
        assert prov == ""
        assert key == ""

    def test_explicit_model_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "k")
        cfg = AIConfig(provider="minimax", model="Custom-Model-1")
        prov, key, model = resolve_active_provider(cfg)
        assert model == "Custom-Model-1"


# ── Retry behaviour ─────────────────────────────────────────────────────────


class TestRetries:
    def test_succeeds_after_transient_5xx(self, monkeypatch: pytest.MonkeyPatch):
        # Patch the actual provider functions to simulate 503 → 200 retries
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "sk-or")
        calls = {"n": 0}
        from app.ai.provider import _openrouter_chat

        def fake_or_chat(messages, model, api_key, timeout, response_format_json=True):
            calls["n"] += 1
            if calls["n"] < 3:
                # raise the same error type the real function would on a 503
                resp = MagicMock()
                resp.status_code = 503
                raise httpx.HTTPStatusError("503", request=MagicMock(), response=resp)
            return {"choices": [{"message": {"content": '{"score": 0.5}'}}]}

        monkeypatch.setattr(provider, "_openrouter_chat", fake_or_chat)
        cfg = AIConfig(provider="openrouter", cache_ttl_seconds=0, max_retries=3)
        result = provider.call_ai("hi", mode="hybrid", cfg=cfg)
        assert result["json"] == {"score": 0.5}
        assert calls["n"] == 3

    def test_gives_up_on_persistent_5xx(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "sk-or")
        from app.ai.provider import _openrouter_chat

        def fake_or_chat(messages, model, api_key, timeout, response_format_json=True):
            resp = MagicMock()
            resp.status_code = 500
            raise httpx.HTTPStatusError("500", request=MagicMock(), response=resp)

        monkeypatch.setattr(provider, "_openrouter_chat", fake_or_chat)
        cfg = AIConfig(provider="openrouter", cache_ttl_seconds=0, max_retries=2)
        with pytest.raises(httpx.HTTPStatusError):
            # Bypass base delay by patching time.sleep
            import time as _t
            monkeypatch.setattr(_t, "sleep", lambda *_: None)
            provider.call_ai("hi", mode="hybrid", cfg=cfg)


# ── call_ai end-to-end (function-level patching) ───────────────────────────


class TestCallAI:
    def test_minimax_uses_anthropic_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "sk-test")
        captured = {}

        def fake_minimax(messages, model, api_key, max_tokens, timeout):
            captured["messages"] = messages
            captured["model"] = model
            captured["api_key"] = api_key
            captured["max_tokens"] = max_tokens
            return {
                "id": "msg_x",
                "content": [{"type": "text", "text": '{"score": 0.42, "rationale": "ok"}'}],
            }

        monkeypatch.setattr(provider, "_minimax_chat", fake_minimax)
        cfg = AIConfig(provider="minimax", cache_ttl_seconds=0, max_retries=0)
        result = provider.call_ai("hello", mode="hybrid", cfg=cfg)
        assert result["provider"] == "minimax"
        assert result["json"] == {"score": 0.42, "rationale": "ok"}
        assert captured["api_key"] == "sk-test"
        assert captured["model"] == MINIMAX_DEFAULT_MODEL
        # Prompt is in messages[0]['content']
        assert any("hello" in (m.get("content") or "") for m in captured["messages"])

    def test_openrouter_uses_bearer_auth(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "")
        monkeypatch.setattr(provider, "get_openrouter_key", lambda: "sk-or")
        captured = {}

        def fake_or(messages, model, api_key, timeout, response_format_json=True):
            captured["api_key"] = api_key
            captured["model"] = model
            return {
                "choices": [{"message": {"content": '{"score": 0.9}'}}]
            }

        monkeypatch.setattr(provider, "_openrouter_chat", fake_or)
        cfg = AIConfig(provider="openrouter", cache_ttl_seconds=0, max_retries=0)
        result = provider.call_ai("hi", mode="hybrid", cfg=cfg)
        assert result["provider"] == "openrouter"
        assert result["json"] == {"score": 0.9}
        assert captured["api_key"] == "sk-or"
        assert captured["model"] == OPENROUTER_DEFAULT_MODEL

    def test_cache_hit_skips_network(self, monkeypatch: pytest.MonkeyPatch):
        cache.clear()
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "sk-test")
        calls = {"n": 0}

        def fake_minimax(messages, model, api_key, max_tokens, timeout):
            calls["n"] += 1
            return {"content": [{"type": "text", "text": '{"score": 0.11}'}]}

        monkeypatch.setattr(provider, "_minimax_chat", fake_minimax)
        cfg = AIConfig(provider="minimax", cache_ttl_seconds=600, max_retries=0)
        r1 = provider.call_ai("dup", mode="hybrid", cfg=cfg)
        r2 = provider.call_ai("dup", mode="hybrid", cfg=cfg)
        assert r1["json"] == r2["json"]
        assert calls["n"] == 1  # second call hit the cache

    def test_require_json_raises_on_unparseable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(provider, "get_minimax_key", lambda: "sk")

        def fake_minimax(messages, model, api_key, max_tokens, timeout):
            return {"content": [{"type": "text", "text": "no json here, sorry"}]}

        monkeypatch.setattr(provider, "_minimax_chat", fake_minimax)
        cfg = AIConfig(provider="minimax", cache_ttl_seconds=0, max_retries=0, require_json=True)
        with pytest.raises(RuntimeError, match="did not contain parseable JSON"):
            provider.call_ai("x", mode="hybrid", cfg=cfg)
