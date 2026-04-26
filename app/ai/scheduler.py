"""Background AI analysis scheduler — runs only when markets are open."""
import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import yaml

from app.ai import score_store
from app.ai.openrouter_client import fetch_ai_score

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_force_event = threading.Event()   # set to trigger an immediate analysis cycle

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "settings.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_api_key() -> str:
    import json as _json
    # 1. data/settings.json (UI-configured, in Docker volume)
    if SETTINGS_PATH.exists():
        try:
            val = _json.loads(SETTINGS_PATH.read_text()).get("OPENROUTER_API_KEY", "")
            if val:
                return val
        except Exception:
            pass
    # 2. docker-compose environment / host env var (survives volume wipes)
    val = os.environ.get("OPENROUTER_API_KEY", "")
    if val:
        return val
    # 3. .env file (local dev)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    return ""


def _any_market_open() -> bool:
    try:
        from app.data.markets import any_market_open
        return any_market_open()
    except Exception:
        return True  # fail open — don't block analysis if check fails


async def _analyze_all(tickers: list[str], cfg: dict) -> None:
    ai_cfg = cfg.get("ai_analysis", {})
    api_key = _get_api_key()
    model = ai_cfg.get("model", "perplexity/sonar")
    timeout = ai_cfg.get("timeout_seconds", 30)

    if not api_key:
        logger.warning("AI scheduler: OPENROUTER_API_KEY not set — skipping")
        return

    sem = asyncio.Semaphore(2)

    async def analyze_one(ticker: str) -> None:
        async with sem:
            try:
                score, rationale, sources = await fetch_ai_score(ticker, model, api_key, timeout)
                score_store.set_score(ticker, score, rationale, sources)
                logger.info("AI score %s → %.2f | %s", ticker, score, rationale[:80])
            except Exception as exc:
                logger.warning("AI score failed for %s: %s", ticker, exc)

    await asyncio.gather(*[analyze_one(t) for t in tickers])


def _run_loop(tickers: list[str]) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not _stop_event.is_set():
            cfg = _load_config()
            ai_cfg = cfg.get("ai_analysis", {})

            if not ai_cfg.get("enabled", False):
                _stop_event.wait(timeout=60)
                continue

            market_open = _any_market_open()
            forced = _force_event.is_set()
            _force_event.clear()

            if market_open or forced:
                if not market_open and forced:
                    logger.info("AI scheduler: forced analysis (market closed)")
                loop.run_until_complete(_analyze_all(tickers, cfg))
            else:
                logger.debug("AI scheduler: markets closed — skipping cycle")

            interval = ai_cfg.get("interval_seconds", 1800)
            # Wake up early if forced while sleeping
            _stop_event.wait(timeout=interval)
    finally:
        loop.close()


def start(tickers: list[str]) -> None:
    """Start the AI scheduler daemon thread."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_run_loop, args=(tickers,), daemon=True, name="ai-scheduler")
    _thread.start()
    logger.info("AI scheduler started for %d tickers", len(tickers))


def force_now() -> None:
    """Trigger an immediate analysis cycle regardless of market hours."""
    _force_event.set()


def stop() -> None:
    """Signal the scheduler to stop."""
    _stop_event.set()
