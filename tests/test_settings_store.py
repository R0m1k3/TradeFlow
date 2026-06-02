"""Tests for the persistent SettingsStore and provider key resolution.

Covers:
* Persistence (write/read/disk round-trip)
* Resolution priority: request_key > store > env > empty
* Thread-safety (concurrent writes don't corrupt the file)
* Provider key integration (Finnhub, Twelve Data, Alpha Vantage)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from app.data.settings_store import (
    KEY_ALPHA_VANTAGE,
    KEY_FINNHUB,
    KEY_OPENROUTER,
    KEY_TWELVE_DATA,
    PROVIDER_KEYS,
    SettingsStore,
    get_store,
)


class TestSettingsStore:
    def setup_method(self) -> None:
        self.tmp = Path("/tmp/test_settings.json")
        if self.tmp.exists():
            self.tmp.unlink()
        self.store = SettingsStore(path=self.tmp)

    def test_set_and_get(self):
        self.store.set("FOO", "bar")
        assert self.store.get("FOO") == "bar"

    def test_empty_string_clears(self):
        self.store.set("FOO", "bar")
        self.store.set("FOO", "")
        assert self.store.get("FOO") == ""
        assert not self.store.has("FOO")

    def test_delete(self):
        self.store.set("FOO", "bar")
        self.store.delete("FOO")
        assert not self.store.has("FOO")

    def test_persists_to_disk(self):
        self.store.set("FOO", "bar")
        # Re-instantiate: should read from disk
        s2 = SettingsStore(path=self.tmp)
        assert s2.get("FOO") == "bar"

    def test_atomic_write_no_corruption(self):
        """The store should never leave a half-written file on disk."""
        self.store.set("A", "1")
        # File should always be valid JSON
        data = json.loads(self.tmp.read_text())
        assert isinstance(data, dict)
        assert data.get("A") == "1"

    def test_concurrent_writes(self):
        """Multiple threads writing different keys shouldn't corrupt the file."""
        errors = []

        def writer(key, value, n):
            try:
                for i in range(n):
                    self.store.set(f"{key}_{i}", f"{value}_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"t{i}", f"v{i}", 20)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        # Verify all writes are present
        for i in range(5):
            for j in range(20):
                assert self.store.get(f"t{i}_{j}") == f"v{i}_{j}"

    def test_all_returns_dict(self):
        self.store.set("A", "1")
        self.store.set("B", "2")
        all_ = self.store.all()
        assert all_ == {"A": "1", "B": "2"}

    def test_provider_keys_canonical(self):
        """Each provider name maps to a well-known canonical key."""
        assert PROVIDER_KEYS["finnhub"] == KEY_FINNHUB == "FINNHUB_API_KEY"
        assert PROVIDER_KEYS["twelve_data"] == KEY_TWELVE_DATA == "TWELVE_DATA_API_KEY"
        assert PROVIDER_KEYS["alpha_vantage"] == KEY_ALPHA_VANTAGE == "ALPHA_VANTAGE_API_KEY"
        assert PROVIDER_KEYS["openrouter"] == KEY_OPENROUTER == "OPENROUTER_API_KEY"


class TestProviderKeyResolution:
    """The 4-level priority: request > store > env > empty."""

    def setup_method(self) -> None:
        self.tmp = Path("/tmp/test_resolve.json")
        if self.tmp.exists():
            self.tmp.unlink()
        self.store = SettingsStore(path=self.tmp)
        # Clean up env vars that might interfere
        for k in [KEY_FINNHUB, KEY_TWELVE_DATA, KEY_ALPHA_VANTAGE, KEY_OPENROUTER]:
            os.environ.pop(k, None)

    def teardown_method(self) -> None:
        for k in [KEY_FINNHUB, KEY_TWELVE_DATA, KEY_ALPHA_VANTAGE, KEY_OPENROUTER]:
            os.environ.pop(k, None)

    def test_request_overrides_store(self):
        self.store.set(KEY_FINNHUB, "from-store")
        assert self.store.get_provider_key("finnhub", request_key="from-request") == "from-request"

    def test_store_overrides_env(self, monkeypatch):
        monkeypatch.setenv(KEY_FINNHUB, "from-env")
        self.store.set(KEY_FINNHUB, "from-store")
        assert self.store.get_provider_key("finnhub") == "from-store"

    def test_env_used_when_store_empty(self, monkeypatch):
        monkeypatch.setenv(KEY_TWELVE_DATA, "from-env")
        assert self.store.get_provider_key("twelve_data") == "from-env"

    def test_empty_when_nothing_configured(self):
        assert self.store.get_provider_key("alpha_vantage") == ""
        assert not self.store.is_provider_configured("alpha_vantage")

    def test_is_provider_configured_uses_request(self, monkeypatch):
        monkeypatch.setenv(KEY_FINNHUB, "from-env")
        # No store key, but request key passed → configured
        assert self.store.is_provider_configured("finnhub", request_key="from-request")


class TestProviderIntegration:
    """End-to-end: a provider reads its key from the SettingsStore."""

    def setup_method(self) -> None:
        # Reset the module-level singleton so each test gets a fresh store
        # pointing at a tmp file. We do this by monkey-patching the path
        # the singleton uses.
        from app.data import settings_store as ss
        ss._default = None
        # We also point the singleton at our tmp file
        self.tmp = Path("/tmp/test_provider_int.json")
        if self.tmp.exists():
            self.tmp.unlink()
        # Reset env vars
        for k in [KEY_FINNHUB, KEY_TWELVE_DATA, KEY_ALPHA_VANTAGE]:
            os.environ.pop(k, None)

    def teardown_method(self) -> None:
        for k in [KEY_FINNHUB, KEY_TWELVE_DATA, KEY_ALPHA_VANTAGE]:
            os.environ.pop(k, None)
        from app.data import settings_store as ss
        ss._default = None

    def _reset_singleton_at(self, path: Path):
        """Force the module-level singleton to use the given path."""
        from app.data import settings_store as ss
        ss._default = SettingsStore(path=path)

    def test_finnhub_reads_from_store(self):
        from app.data.providers.finnhub_provider import FinnhubProvider
        self._reset_singleton_at(self.tmp)
        from app.data.settings_store import get_store
        get_store().set(KEY_FINNHUB, "key-from-store")
        p = FinnhubProvider()
        assert p.is_available()
        assert p._key() == "key-from-store"

    def test_twelve_data_reads_from_store(self):
        from app.data.providers.twelve_data_provider import TwelveDataProvider
        self._reset_singleton_at(self.tmp)
        from app.data.settings_store import get_store
        get_store().set(KEY_TWELVE_DATA, "twelve-key")
        p = TwelveDataProvider()
        assert p.is_available()
        assert p._key() == "twelve-key"

    def test_alpha_vantage_reads_from_store(self):
        from app.data.providers.alpha_vantage_provider import AlphaVantageProvider
        self._reset_singleton_at(self.tmp)
        from app.data.settings_store import get_store
        get_store().set(KEY_ALPHA_VANTAGE, "av-key")
        p = AlphaVantageProvider()
        assert p.is_available()
        assert p._key() == "av-key"

    def test_request_key_overrides_store(self):
        from app.data.providers.finnhub_provider import FinnhubProvider
        self._reset_singleton_at(self.tmp)
        from app.data.settings_store import get_store
        get_store().set(KEY_FINNHUB, "store-key")
        p = FinnhubProvider()
        p.set_request_key("request-key")
        assert p._key() == "request-key"
        p.clear_request_key()
        assert p._key() == "store-key"

    def test_yahoo_always_available(self):
        from app.data.providers.yahoo_provider import YahooProvider
        p = YahooProvider()
        assert p.is_available()  # No key required

    def test_unconfigured_provider_not_available(self):
        from app.data.providers.finnhub_provider import FinnhubProvider
        self._reset_singleton_at(self.tmp)
        from app.data.settings_store import get_store
        # Make sure no key is set
        for k in [KEY_FINNHUB]:
            get_store().delete(k)
        p = FinnhubProvider()
        assert not p.is_available()
        assert p._key() == ""
