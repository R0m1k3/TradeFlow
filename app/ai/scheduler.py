"""Background AI analysis scheduler — runs only when markets are open."""
import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import yaml

from app.ai import score_store
from app.ai.openrouter_client import fetch_ai_score, AUTONOMOUS_PROMPT_TEMPLATE
from app.ai.persist import save_ai_signal

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_force_event = threading.Event()

# Last-run tracking (updated after each completed cycle)
_last_run_at: float | None = None
_last_run_mode: str | None = None
_last_run_ticker_count: int = 0
_status_lock = threading.Lock()


def get_status() -> dict:
    """Return info about the last completed AI analysis cycle."""
    with _status_lock:
        return {
            "last_run_at": _last_run_at,
            "last_run_mode": _last_run_mode,
            "last_run_ticker_count": _last_run_ticker_count,
            "running": _thread is not None and _thread.is_alive(),
        }

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "settings.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_api_key() -> str:
    import json as _json
    # 1. UI-configured, persisted in volume
    if SETTINGS_PATH.exists():
        try:
            val = _json.loads(SETTINGS_PATH.read_text()).get("OPENROUTER_API_KEY", "")
            if val:
                return val
        except Exception:
            pass
    # 2. docker-compose environment / host env var
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


def _get_all_tickers() -> list[str]:
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
            return float(hist.iloc[-1]["Close"])
    except Exception:
        pass
    return None


def _any_market_open() -> bool:
    try:
        from app.data.markets import any_market_open
        return any_market_open()
    except Exception:
        return True


# ── Hybrid mode ───────────────────────────────────────────────────────────────

async def _analyze_all_hybrid(tickers: list[str], cfg: dict) -> None:
    """Score each ticker (0-1) and store in score_store."""
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
                save_ai_signal(
                    symbol=ticker,
                    mode="hybrid",
                    computed_at=datetime.now(timezone.utc),
                    score=score,
                    rationale=rationale,
                    sources=sources,
                )
                logger.info("AI score %s → %.2f | %s", ticker, score, rationale[:80])
            except Exception as exc:
                logger.warning("AI score failed for %s: %s", ticker, exc)

    await asyncio.gather(*[analyze_one(t) for t in tickers])


# ── Autonomous mode ───────────────────────────────────────────────────────────

def _analyze_all_autonomous(tickers: list[str], cfg: dict) -> None:
    """Full autonomous ARIA analysis — BUY/SELL/HOLD with risk params per ticker."""
    import requests as req
    import json as _json
    from datetime import datetime as _dt

    ai_cfg = cfg.get("ai_analysis", {})
    api_key = _get_api_key()
    model = ai_cfg.get("model", "perplexity/sonar")
    timeout = ai_cfg.get("timeout_seconds", 45)

    if not api_key:
        logger.warning("ARIA scheduler: OPENROUTER_API_KEY not set — skipping")
        return

    logger.info("ARIA scheduler: starting autonomous analysis for %d tickers", len(tickers))

    # Sync initial capital from config
    try:
        from app.ai import aria_portfolio as ap
        initial = cfg.get("aria_capital", cfg.get("default_capital", 10000))
        ap.set_initial_capital(float(initial))
    except Exception:
        ap = None

    prices_seen: dict[str, float] = {}

    for ticker in tickers:
        if _stop_event.is_set():
            break
        try:
            prompt = AUTONOMOUS_PROMPT_TEMPLATE.format(
                ticker=ticker,
                date=_dt.now().strftime("%Y-%m-%d %H:%M"),
            )
            r = req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://tradeflow.local",
                    "X-Title": "TradeFlow",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = _json.loads(r.json()["choices"][0]["message"]["content"])

            action = data.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            if confidence < 0.5:
                action = "HOLD"

            pos_pct = max(0.0, min(10.0, float(data.get("position_size_pct", 0))))
            sl = max(0.5, min(15.0, float(data.get("stop_loss_pct", 2.0))))
            tp = max(1.0, min(30.0, float(data.get("take_profit_pct", 4.0))))
            sources = data.get("sources", [])
            if not isinstance(sources, list):
                sources = []

            score_store.set_decision(
                ticker, action, confidence, pos_pct, sl, tp,
                data.get("time_horizon", "24h"),
                data.get("rationale", ""),
                data.get("key_risks", ""),
                sources,
            )
            save_ai_signal(
                symbol=ticker,
                mode="autonomous",
                computed_at=datetime.now(timezone.utc),
                action=action,
                confidence=confidence,
                position_size_pct=pos_pct,
                stop_loss_pct=sl,
                take_profit_pct=tp,
                rationale=data.get("rationale", ""),
                key_risks=data.get("key_risks", ""),
                sources=sources,
            )
            logger.info("ARIA %s → %s (conf=%.2f)", ticker, action, confidence)

            # Execute against ARIA virtual portfolio
            if ap and action in ("BUY", "SELL"):
                price = _fetch_price(ticker)
                if price:
                    prices_seen[ticker] = price
                    ap.execute_decision(ticker, action, pos_pct, sl, tp, price)

        except Exception as exc:
            logger.warning("ARIA analysis failed for %s: %s", ticker, exc)

    # Check SL/TP on all open positions and record snapshot
    if ap:
        try:
            # Fetch prices for any open positions we didn't see this cycle
            for sym in ap.get_open_positions():
                if sym not in prices_seen:
                    p = _fetch_price(sym)
                    if p:
                        prices_seen[sym] = p
            ap.check_stops(prices_seen)
            ap.take_snapshot(prices_seen)
        except Exception as exc:
            logger.warning("ARIA portfolio snapshot failed: %s", exc)


# ── Main loop ─────────────────────────────────────────────────────────────────

def _run_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not _stop_event.is_set():
            cfg = _load_config()
            ai_cfg = cfg.get("ai_analysis", {})

            if not ai_cfg.get("enabled", False):
                _stop_event.wait(timeout=60)
                continue

            tickers = _get_all_tickers()
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

                mode = ai_cfg.get("mode", "hybrid")
                if mode == "autonomous":
                    _analyze_all_autonomous(tickers, cfg)
                else:
                    loop.run_until_complete(_analyze_all_hybrid(tickers, cfg))

                with _status_lock:
                    global _last_run_at, _last_run_mode, _last_run_ticker_count
                    _last_run_at = time.time()
                    _last_run_mode = mode
                    _last_run_ticker_count = len(tickers)
                logger.info("AI scheduler: cycle complete (%s, %d tickers)", mode, len(tickers))
            else:
                logger.debug("AI scheduler: markets closed — skipping cycle")

            interval = ai_cfg.get("interval_seconds", 1800)
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
