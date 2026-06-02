"""Background AI analysis scheduler — runs only when markets are open.

Provider-agnostic: routes both hybrid (sentiment score) and autonomous
(BUY/SELL/HOLD with risk params) analyses through `app.ai.provider.call_ai`.

Improvements over the previous version:
  * Unified provider with retries, caching, and JSON-fence stripping.
  * Increased parallelism (semaphore=4) for hybrid mode.
  * Autonomous mode also parallelised via a small thread pool — the bottleneck
    is the upstream LLM, not the local loop.
  * Hardened stop-event handling so `force_now()` interrupts mid-batch.
  * Emits structured per-cycle stats to the in-memory `status` for the UI.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from app.ai import cache, score_store
from app.ai.openrouter_client import AUTONOMOUS_PROMPT_TEMPLATE
from app.ai.persist import save_ai_signal
from app.ai.provider import (
    AIConfig,
    call_ai,
    load_ai_config,
    resolve_active_provider,
)

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_force_event = threading.Event()

# Last-run tracking (updated after each completed cycle)
_last_run_at: float | None = None
_last_run_mode: str | None = None
_last_run_provider: str | None = None
_last_run_model: str | None = None
_last_run_ticker_count: int = 0
_last_run_success_count: int = 0
_last_run_error_count: int = 0
_status_lock = threading.Lock()


def get_status() -> dict:
    """Return info about the last completed AI analysis cycle."""
    with _status_lock:
        return {
            "last_run_at": _last_run_at,
            "last_run_mode": _last_run_mode,
            "last_run_provider": _last_run_provider,
            "last_run_model": _last_run_model,
            "last_run_ticker_count": _last_run_ticker_count,
            "last_run_success_count": _last_run_success_count,
            "last_run_error_count": _last_run_error_count,
            "running": _thread is not None and _thread.is_alive(),
        }


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "settings.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_api_tickers() -> list[str]:
    try:
        from app.data.nasdaq import get_all_tickers
        return get_all_tickers()
    except Exception as exc:
        logger.warning("Could not load ticker list: %s", exc)
        return []


def _fetch_price(symbol: str) -> float | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1d", interval="5m")
        if hist is not None and not hist.empty:
            close_col = "Close" if "Close" in hist.columns else "close"
            return float(hist.iloc[-1][close_col])
    except Exception:
        pass
    return None


def _any_market_open() -> bool:
    try:
        from app.data.markets import any_market_open
        return any_market_open()
    except Exception:
        return True


# ── Hybrid mode ─────────────────────────────────────────────────────────────


async def _analyze_one_hybrid(ticker: str, cfg: AIConfig) -> tuple[str, Optional[dict], Optional[str]]:
    """Score a single ticker through the unified provider."""
    from app.ai.openrouter_client import PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker, date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    try:
        result = await asyncio.to_thread(call_ai, prompt, mode="hybrid", cfg=cfg)
    except Exception as exc:
        return ticker, None, str(exc)
    parsed = result.get("json") or {}
    try:
        score = float(parsed.get("score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    rationale = parsed.get("rationale", "") or ""
    sources = parsed.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return ticker, {"score": score, "rationale": rationale, "sources": sources,
                    "provider": result["provider"], "model": result["model"]}, None


async def _analyze_all_hybrid(tickers: list[str], cfg: AIConfig) -> tuple[int, int]:
    """Score each ticker (0-1) and store in score_store. Returns (success, errors)."""
    sem = asyncio.Semaphore(4)

    async def runner(t: str):
        async with sem:
            return await _analyze_one_hybrid(t, cfg)

    tasks = [runner(t) for t in tickers if not _stop_event.is_set()]
    if not tasks:
        return 0, 0
    results = await asyncio.gather(*tasks, return_exceptions=False)

    success = 0
    errors = 0
    for ticker, payload, err in results:
        if err or payload is None:
            errors += 1
            logger.warning("AI score failed for %s: %s", ticker, err)
            continue
        score_store.set_score(
            ticker, payload["score"], payload["rationale"], payload["sources"]
        )
        try:
            save_ai_signal(
                symbol=ticker,
                mode="hybrid",
                computed_at=datetime.now(timezone.utc),
                score=payload["score"],
                rationale=payload["rationale"],
                sources=payload["sources"],
            )
        except Exception as exc:
            logger.debug("Persist hybrid signal failed for %s: %s", ticker, exc)
        success += 1
        logger.info(
            "AI score %s → %.2f via %s/%s | %s",
            ticker, payload["score"], payload["provider"], payload["model"],
            payload["rationale"][:80],
        )
    return success, errors


# ── Autonomous mode ─────────────────────────────────────────────────────────


def _analyze_one_autonomous_sync(ticker: str, cfg: AIConfig) -> dict:
    """Run a single autonomous analysis synchronously and return the decision."""
    prompt = AUTONOMOUS_PROMPT_TEMPLATE.format(
        ticker=ticker,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    result = call_ai(prompt, mode="autonomous", cfg=cfg)
    data = result.get("json") or {}

    action = (data.get("action") or "HOLD").upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    if confidence < 0.5:
        action = "HOLD"
    pos_pct = max(0.0, min(10.0, float(data.get("position_size_pct", 0) or 0)))
    sl = max(0.5, min(15.0, float(data.get("stop_loss_pct", 2.0) or 2.0)))
    tp = max(1.0, min(30.0, float(data.get("take_profit_pct", 4.0) or 4.0)))
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    return {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "position_size_pct": pos_pct,
        "stop_loss_pct": sl,
        "take_profit_pct": tp,
        "time_horizon": data.get("time_horizon", "24h"),
        "rationale": data.get("rationale", "") or "",
        "key_risks": data.get("key_risks", "") or "",
        "sources": sources,
        "provider": result["provider"],
        "model": result["model"],
        "error": None,
    }


def _analyze_all_autonomous(tickers: list[str], cfg: AIConfig) -> tuple[int, int, dict[str, float]]:
    """Full autonomous ARIA analysis — BUY/SELL/HOLD with risk params per ticker."""
    provider, _, model = resolve_active_provider(cfg)
    if not provider:
        logger.warning("ARIA scheduler: no AI provider key configured — skipping")
        return 0, 0, {}
    logger.info(
        "ARIA scheduler: autonomous analysis for %d tickers via %s/%s",
        len(tickers), provider, model,
    )

    # Sync initial capital from config (read from YAML; not part of AIConfig)
    ap = None
    try:
        from app.ai import aria_portfolio as _ap
        import yaml as _yaml
        yaml_cfg = _yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        capital = float(yaml_cfg.get("aria_capital", 10000))
        _ap.set_initial_capital(capital)
        ap = _ap
    except Exception as exc:
        logger.debug("ARIA capital init failed: %s", exc)

    prices_seen: dict[str, float] = {}
    success = 0
    errors = 0
    decisions: dict[str, dict] = {}

    # Parallelise across tickers (LLM call is the bottleneck)
    max_workers = min(4, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_analyze_one_autonomous_sync, t, cfg): t
            for t in tickers
            if not _stop_event.is_set()
        }
        for fut in as_completed(futures):
            if _stop_event.is_set():
                break
            ticker = futures[fut]
            try:
                d = fut.result()
            except Exception as exc:
                errors += 1
                logger.warning("ARIA analysis failed for %s: %s", ticker, exc)
                continue
            if d.get("error"):
                errors += 1
                continue
            decisions[ticker] = d
            score_store.set_decision(
                ticker, d["action"], d["confidence"], d["position_size_pct"],
                d["stop_loss_pct"], d["take_profit_pct"], d["time_horizon"],
                d["rationale"], d["key_risks"], d["sources"],
            )
            try:
                save_ai_signal(
                    symbol=ticker,
                    mode="autonomous",
                    computed_at=datetime.now(timezone.utc),
                    action=d["action"],
                    confidence=d["confidence"],
                    position_size_pct=d["position_size_pct"],
                    stop_loss_pct=d["stop_loss_pct"],
                    take_profit_pct=d["take_profit_pct"],
                    rationale=d["rationale"],
                    key_risks=d["key_risks"],
                    sources=d["sources"],
                )
            except Exception as exc:
                logger.debug("Persist autonomous signal failed for %s: %s", ticker, exc)
            success += 1
            logger.info("ARIA %s → %s (conf=%.2f) via %s", ticker, d["action"],
                        d["confidence"], d["provider"])

            if d["action"] in ("BUY", "SELL") and ap is not None:
                try:
                    price = _fetch_price(ticker)
                    if price:
                        prices_seen[ticker] = price
                        ap.execute_decision(
                            ticker, d["action"], d["position_size_pct"],
                            d["stop_loss_pct"], d["take_profit_pct"], price,
                        )
                except Exception as exc:
                    logger.warning("ARIA portfolio exec failed for %s: %s", ticker, exc)

    # Check SL/TP on all open positions and record snapshot
    if ap is not None:
        try:
            for sym in ap.get_open_positions():
                if sym not in prices_seen:
                    p = _fetch_price(sym)
                    if p:
                        prices_seen[sym] = p
            ap.check_stops(prices_seen)
            ap.take_snapshot(prices_seen)
        except Exception as exc:
            logger.warning("ARIA portfolio snapshot failed: %s", exc)

    return success, errors, decisions


# ── Main loop ───────────────────────────────────────────────────────────────


def _run_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not _stop_event.is_set():
            cfg_dict = _load_config()
            ai_cfg_dict = cfg_dict.get("ai_analysis", {})

            if not ai_cfg_dict.get("enabled", False):
                _stop_event.wait(timeout=60)
                continue

            tickers = _get_api_tickers()
            if not tickers:
                logger.warning("AI scheduler: no tickers found — retrying in 60s")
                _stop_event.wait(timeout=60)
                continue

            market_open = _any_market_open()
            forced = _force_event.is_set()
            _force_event.clear()

            if market_open or forced:
                if not market_open and forced:
                    logger.info("AI scheduler: forced analysis (market closed)")

                cfg = load_ai_config(ai_cfg_dict)
                mode = ai_cfg_dict.get("mode", "hybrid")

                # Resolve active provider once per cycle for status reporting
                provider, _key, model = resolve_active_provider(cfg)
                if not provider:
                    logger.warning("AI scheduler: no provider available (check API keys)")
                    _stop_event.wait(timeout=300)
                    continue

                if mode == "autonomous":
                    success, errors, _ = _analyze_all_autonomous(tickers, cfg)
                else:
                    success, errors = loop.run_until_complete(
                        _analyze_all_hybrid(tickers, cfg)
                    )

                with _status_lock:
                    global _last_run_at, _last_run_mode, _last_run_provider
                    global _last_run_model, _last_run_ticker_count
                    global _last_run_success_count, _last_run_error_count
                    _last_run_at = time.time()
                    _last_run_mode = mode
                    _last_run_provider = provider
                    _last_run_model = model
                    _last_run_ticker_count = len(tickers)
                    _last_run_success_count = success
                    _last_run_error_count = errors
                logger.info(
                    "AI scheduler: cycle complete (%s, %d tickers, %d ok / %d errors, %s/%s)",
                    mode, len(tickers), success, errors, provider, model,
                )
            else:
                logger.debug("AI scheduler: markets closed — skipping cycle")

            interval = ai_cfg_dict.get("interval_seconds", 1800)
            _stop_event.wait(timeout=interval)
    finally:
        loop.close()


def start() -> None:
    """Start the AI scheduler daemon thread (always — checks enabled flag internally)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _force_event.set()  # burst initial
    _thread = threading.Thread(target=_run_loop, daemon=True, name="ai-scheduler")
    _thread.start()
    logger.info("AI scheduler started (with immediate burst)")


def force_now() -> None:
    """Trigger an immediate analysis cycle regardless of market hours."""
    _force_event.set()


def stop() -> None:
    """Signal the scheduler to stop."""
    _stop_event.set()


def cache_stats() -> dict:
    """Expose the persistent cache stats for the UI."""
    return cache.stats()
