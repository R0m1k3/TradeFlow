"""
TradeFlow — FastAPI Server
Serves the React SPA and provides REST API endpoints backed by the existing Python modules.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────────────────
# Add project root so we can import app.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
import queue

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analysis.composite import compute_composite_score
from app.data.finnhub_client import subscribe_symbols, add_subscriber, remove_subscriber, get_all_prices
from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.data.markets import (
    EXCHANGES,
    any_market_open,
    get_all_market_statuses,
    is_market_open,
    next_market_event,
)
from app.data.nasdaq import (
    get_all_tickers,
    get_display_name,
    get_currency,
    search_tickers,
    STOCK_INFO,
)
from app.database.session import get_session, init_database, create_db_engine
from app.database.models import Base, Portfolio as PortfolioModel, SimRun, Trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradeflow.server")

# ── FastAPI app ───────────────────────────────────────────────────────────────────

app = FastAPI(title="TradeFlow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
HTML_FILE = HERE / "TradeFlow.html"
BOT_PID_FILE = _PROJECT_ROOT / "data" / "bot.pid"
CONFIG_FILE = _PROJECT_ROOT / "config.yaml"
ENV_FILE = _PROJECT_ROOT / ".env"
DATA_DIR = _PROJECT_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"   # persistent across Docker restarts

DATA_DIR.mkdir(parents=True, exist_ok=True)

init_database()

# ── AI Scheduler startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def _start_ai_scheduler():
    """Start the AI analysis scheduler on server startup."""
    try:
        from app.ai import scheduler as ai_scheduler
        ai_scheduler.start()
        logger.info("AI scheduler launched from server startup")
    except Exception as exc:
        logger.warning("Could not start AI scheduler: %s", exc)

# ── Helpers ───────────────────────────────────────────────────────────────────────


def _load_settings() -> dict:
    """Load persistent user settings from data/settings.json (survives Docker restarts)."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _read_env_key(name: str) -> str:
    """Read a key: settings.json → env var (docker-compose) → .env file."""
    # 1. UI-configured, persisted in volume (survives restarts, lost only on volume wipe)
    val = _load_settings().get(name, "")
    if val:
        return val
    # 2. docker-compose environment / host env var (survives volume wipes)
    val = os.environ.get(name, "")
    if val:
        return val
    # 3. .env file (local dev convenience)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def _write_env_key(name: str, value: str) -> None:
    """Persist a key to settings.json (Docker-safe) and .env (local dev)."""
    settings = _load_settings()
    if value:
        settings[name] = value
    else:
        settings.pop(name, None)
    _save_settings(settings)
    # Also write to .env for local dev convenience
    if ENV_FILE.exists() or not DATA_DIR.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
        lines = [l for l in lines if not l.startswith(f"{name}=")]
        if value:
            lines.append(f"{name}={value}")
        try:
            ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass


def _load_config() -> dict:
    """Load YAML config, return empty dict on failure."""
    try:
        import yaml

        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load config: %s", exc)
    return {}


def _get_bot_pid() -> int | None:
    """Read the bot PID file. Returns None if missing or stale."""
    if not BOT_PID_FILE.exists():
        return None
    try:
        pid = int(BOT_PID_FILE.read_text(encoding="utf-8").strip())
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
            )
            if str(pid) not in result.stdout:
                BOT_PID_FILE.unlink(missing_ok=True)
                return None
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                BOT_PID_FILE.unlink(missing_ok=True)
                return None
        return pid
    except (ValueError, OSError):
        BOT_PID_FILE.unlink(missing_ok=True)
        return None


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}"


def _compute_signal(score: float) -> str:
    if score >= 0.7:
        return "buy"
    if score <= 0.3:
        return "sell"
    return "watch"


def _compute_sparkline(prices: list[float], n: int = 32) -> list[float]:
    """Downsample a price list to n points for sparkline display."""
    if not prices:
        return []
    if len(prices) <= n:
        return prices
    step = len(prices) / n
    result = []
    for i in range(n):
        idx = int(i * step)
        result.append(float(prices[min(idx, len(prices) - 1)]))
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════


@app.get("/api/markets")
def get_markets():
    """Return exchange / index data with current values and market status."""
    results = []
    for ex in EXCHANGES:
        open_now = is_market_open(ex)
        val = None
        chg = None
        try:
            idx = yf.Ticker(ex.index_ticker)
            hist = idx.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                # Normalize column names (yfinance uses PascalCase by default)
                close_col = "Close" if "Close" in hist.columns else "close"
                val = float(hist.iloc[-1][close_col])
                if len(hist) >= 2:
                    prev = float(hist.iloc[-2][close_col])
                    chg = round(((val - prev) / prev) * 100, 2)
        except Exception as exc:
            logger.warning("Failed to fetch index %s: %s", ex.index_ticker, exc, exc_info=True)

        results.append({
            "exch": ex.name.upper(),
            "name": ex.index_name,
            "ticker": ex.index_ticker,
            "val": val,
            "chg": chg,
            "open": open_now,
        })

    return {
        "exchanges": results,
        "any_open": any(r["open"] for r in results),
        "next_event": str(next_market_event()) if not any(r["open"] for r in results) else None,
    }


@app.get("/api/stocks")
def get_stocks(
    query: str = Query("", description="Search query"),
    sort_by: str = Query("score", description="Sort field"),
    sort_dir: str = Query("desc", description="Sort direction"),
    limit: int = Query(200, description="Max results"),
):
    """Return stock list with scores, signals, and sparklines."""
    if query:
        symbols = search_tickers(query, limit=limit)
    else:
        symbols = get_all_tickers()[:limit]

    interval = "1d"
    period = "3mo"

    def _fetch_symbol(sym: str):
        try:
            df = fetch_ohlcv(sym, interval=interval, period=period)
            if df is None or df.empty:
                return None, None
            df = add_all_indicators(df)
            df.attrs["symbol"] = sym
            score_data = compute_composite_score(df, sym)
            price = float(df.iloc[-1]["close"])
            if len(df) >= 2:
                prev_close = float(df.iloc[-2]["close"])
                chg = round(((price - prev_close) / prev_close) * 100, 2)
            else:
                chg = 0.0
            prices = df["close"].tolist()
            sparkline = _compute_sparkline(prices, 32)
            sig = _compute_signal(score_data.combined)
            name, curr = STOCK_INFO.get(sym, (sym, "USD"))
            return {
                "sym": sym,
                "name": name,
                "price": round(price, 2),
                "chg": chg,
                "score": round(score_data.combined, 2),
                "signal": sig,
                "currency": curr,
                "sparkline": [round(float(v), 2) for v in sparkline],
            }, None
        except Exception as exc:
            return None, {"sym": sym, "error": str(exc)}

    stocks = []
    errors = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            result, err = fut.result()
            if result:
                stocks.append(result)
            elif err:
                errors.append(err)

    # Sort
    reverse = sort_dir.lower() != "asc"
    stocks.sort(key=lambda s: s.get(sort_by, 0) if isinstance(s.get(sort_by), (int, float)) else str(s.get(sort_by, "")), reverse=reverse)

    return {"stocks": stocks, "total": len(stocks), "errors": errors[:5]}


@app.get("/api/stocks/{symbol}")
def get_stock_detail(symbol: str):
    """Return full detail for a single stock including stats and history."""
    try:
        df = fetch_ohlcv(symbol, interval="1d", period="1y")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        df = add_all_indicators(df)
        df.attrs["symbol"] = symbol
        score_data = compute_composite_score(df, symbol)

        price = float(df.iloc[-1]["close"])
        if len(df) >= 2:
            prev_close = float(df.iloc[-2]["close"])
            chg = round(((price - prev_close) / prev_close) * 100, 2)
        else:
            chg = 0.0

        high = float(df["high"].max())
        low = float(df["low"].min())
        volume = int(df["volume"].sum())
        avg_volume = int(df["volume"].tail(30).mean()) if len(df) >= 30 else volume

        name, curr = STOCK_INFO.get(symbol, (symbol, "USD"))

        # Determine exchange
        exchange = "NASDAQ"
        if ".PA" in symbol:
            exchange = "Euronext Paris"
        elif ".DE" in symbol:
            exchange = "Xetra Frankfurt"

        sig = _compute_signal(score_data.combined)

        stats = {
            "open": round(float(df.iloc[-1]["open"]), 2),
            "high_52w": round(high, 2),
            "low_52w": round(low, 2),
            "volume": volume,
            "avg_volume_30d": avg_volume,
            "rsi": round(float(score_data.rsi_score * 100), 1),
            "score": round(score_data.combined, 2),
            "confidence": round(score_data.combined * 100, 0),
        }

        # Historical prices for chart
        history = [
            {"date": str(idx.date()), "price": round(float(row["close"]), 2)}
            for idx, row in df.iterrows()
        ]

        return {
            "sym": symbol,
            "name": name,
            "price": round(price, 2),
            "chg": chg,
            "score": round(score_data.combined, 2),
            "signal": sig,
            "currency": curr,
            "exchange": exchange,
            "stats": stats,
            "history": history,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to fetch detail for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stocks/{symbol}/history")
def get_stock_history(
    symbol: str,
    range: str = Query("1M", description="Time range: 1J, 1S, 1M, 3M, 1A, TOUT"),
):
    """Return price history for charting."""
    range_map = {
        "1J": ("5m", "1d"),
        "1S": ("1h", "5d"),
        "1M": ("1d", "1mo"),
        "3M": ("1d", "3mo"),
        "1A": ("1d", "1y"),
        "TOUT": ("1d", "max"),
    }
    interval, period = range_map.get(range, ("1d", "3mo"))

    try:
        df = fetch_ohlcv(symbol, interval=interval, period=period)
        if df is None or df.empty:
            return {"dates": [], "prices": [], "volumes": []}

        return {
            "dates": [str(idx.date()) for idx in df.index],
            "prices": [round(float(v), 2) for v in df["close"].tolist()],
            "volumes": [int(v) for v in df["volume"].tolist()],
        }
    except Exception as exc:
        logger.error("Failed to fetch history for %s: %s", symbol, exc)
        return {"dates": [], "prices": [], "volumes": [], "error": str(exc)}


@app.get("/api/bot/status")
def get_bot_status():
    """Return the current bot status."""
    pid = _get_bot_pid()
    uptime = None
    trades_today = 0

    if pid is not None:
        try:
            with get_session() as session:
                today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                trades_today = (
                    session.query(Trade)
                    .filter(Trade.timestamp >= today_start)
                    .count()
                )
        except Exception:
            pass

        # Estimate uptime from PID file mtime
        try:
            mtime = BOT_PID_FILE.stat().st_mtime
            uptime_seconds = int(datetime.now(timezone.utc).timestamp() - mtime)
            uptime = str(timedelta(seconds=uptime_seconds))
        except Exception:
            pass

    return {
        "active": pid is not None,
        "pid": pid,
        "uptime": uptime,
        "trades_today": trades_today,
    }


@app.post("/api/bot/start")
def start_bot():
    """Start the live trading bot as a background process."""
    if _get_bot_pid() is not None:
        return {"success": True, "message": "Bot already running"}

    bot_script = _PROJECT_ROOT / "app" / "bot" / "run_bot.py"
    log_file = DATA_DIR / "bot.log"

    try:
        proc = subprocess.Popen(
            [sys.executable, str(bot_script)],
            stdout=open(log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        BOT_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        logger.info("Bot started with PID %s", proc.pid)
        return {"success": True, "pid": proc.pid, "message": "Bot started"}
    except Exception as exc:
        logger.error("Failed to start bot: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/bot/stop")
def stop_bot():
    """Stop the live trading bot."""
    pid = _get_bot_pid()
    if pid is None:
        return {"success": True, "message": "Bot not running"}

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        BOT_PID_FILE.unlink(missing_ok=True)
        logger.info("Bot stopped (PID %s)", pid)
        return {"success": True, "message": "Bot stopped"}
    except Exception as exc:
        logger.error("Failed to stop bot: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/config")
def get_config():
    """Load current configuration."""
    cfg = _load_config()

    lt = cfg.get("live_trading", {})
    default_symbols = cfg.get("default_symbols", ["AAPL", "MSFT", "NVDA"])

    ai = cfg.get("ai_analysis", {})
    saved_key = _read_env_key("OPENROUTER_API_KEY")

    return {
        "capital": lt.get("initial_capital", cfg.get("algo_capital", cfg.get("default_capital", 10000))),
        "algo_capital": cfg.get("algo_capital", cfg.get("default_capital", 10000)),
        "aria_capital": cfg.get("aria_capital", 10000),
        "risk": 2.0,
        "max_positions": len(default_symbols),
        "strategy": lt.get("strategy", "composite"),
        "strategy_label": {
            "composite": "Équilibrée",
            "sma_crossover": "Prudente",
            "rsi": "Prudente",
            "macd": "Agressive",
        }.get(lt.get("strategy", ""), "Équilibrée"),
        "min_score": 65,
        "auto_buy": True,
        "auto_sell": True,
        "notifications": True,
        "symbols": lt.get("symbols", default_symbols),
        "interval": lt.get("interval", "1h"),
        # AI analysis
        "ai_enabled": ai.get("enabled", False),
        "ai_mode": ai.get("mode", "hybrid"),
        "ai_model": ai.get("model", "perplexity/sonar"),
        "ai_score_weight": ai.get("score_weight", 0.3),
        "ai_key_configured": bool(saved_key),
        "ai_key_hint": f"{saved_key[:8]}••••" if saved_key else "",
    }


@app.post("/api/config")
def save_config(body: dict = Body(default={})):
    """Save configuration to config.yaml."""
    try:
        import yaml

        cfg = _load_config()
        lt = cfg.setdefault("live_trading", {})

        if "algo_capital" in body:
            cfg["algo_capital"] = float(body["algo_capital"])
            lt["initial_capital"] = float(body["algo_capital"])
        elif "capital" in body:
            lt["initial_capital"] = body["capital"]
        if "aria_capital" in body:
            cfg["aria_capital"] = float(body["aria_capital"])
            try:
                from app.ai import aria_portfolio as ap
                ap.set_initial_capital(float(body["aria_capital"]))
            except Exception:
                pass
        if "strategy" in body:
            lt["strategy"] = body["strategy"]
            strategy_label_map = {
                "Prudente": "sma_crossover",
                "Équilibrée": "composite",
                "Agressive": "macd",
            }
            if body["strategy"] in strategy_label_map:
                lt["strategy"] = strategy_label_map[body["strategy"]]
        if "symbols" in body:
            lt["symbols"] = body["symbols"]
        if "interval" in body:
            lt["interval"] = body["interval"]
        if "min_score" in body:
            cfg["min_score"] = body["min_score"]
        if "auto_buy" in body:
            cfg["auto_buy"] = body["auto_buy"]
        if "auto_sell" in body:
            cfg["auto_sell"] = body["auto_sell"]
        if "notifications" in body:
            cfg["notifications"] = body["notifications"]
        if "max_positions" in body:
            cfg["max_positions"] = body["max_positions"]

        # AI analysis settings
        ai = cfg.setdefault("ai_analysis", {})
        if "ai_enabled" in body:
            ai["enabled"] = bool(body["ai_enabled"])
        if "ai_mode" in body and body["ai_mode"] in ("hybrid", "autonomous"):
            ai["mode"] = body["ai_mode"]
        if "ai_model" in body:
            ai["model"] = str(body["ai_model"])
        if "ai_score_weight" in body:
            ai["score_weight"] = float(body["ai_score_weight"])
        if "ai_key" in body and body["ai_key"]:
            _write_env_key("OPENROUTER_API_KEY", str(body["ai_key"]))

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        logger.info("Configuration saved")
        return {"success": True, "message": "Configuration saved"}
    except Exception as exc:
        logger.error("Failed to save config: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/portfolio")
def get_portfolio():
    """Return portfolio positions and performance."""
    try:
        with get_session() as session:
            # Get latest portfolio entry
            portfolio = (
                session.query(PortfolioModel)
                .order_by(PortfolioModel.sim_run_id.desc())
                .first()
            )

            # Get recent trades
            recent_trades = (
                session.query(Trade)
                .order_by(Trade.timestamp.desc())
                .limit(20)
                .all()
            )

            # Get latest sim run
            latest_run = (
                session.query(SimRun)
                .order_by(SimRun.id.desc())
                .first()
            )

            positions = []
            if portfolio is not None:
                positions = [
                    {
                        "symbol": p.symbol if hasattr(p, "symbol") else "?",
                        "quantity": p.quantity if hasattr(p, "quantity") else 0,
                        "avg_price": p.avg_buy_price if hasattr(p, "avg_buy_price") else 0,
                    }
                    for p in [portfolio]  # Simplified; real position loading depends on schema
                ]

            trades = [
                {
                    "time": str(t.timestamp) if hasattr(t, "timestamp") else "",
                    "side": t.side if hasattr(t, "side") else "",
                    "symbol": t.symbol if hasattr(t, "symbol") else "",
                    "quantity": float(t.quantity) if hasattr(t, "quantity") else 0,
                    "price": float(t.price) if hasattr(t, "price") else 0,
                }
                for t in recent_trades
            ]

            return {
                "positions": positions,
                "trades": trades,
                "total_value": float(latest_run.final_value) if latest_run and hasattr(latest_run, "final_value") else None,
                "total_return_pct": float(latest_run.total_return_pct) if latest_run and hasattr(latest_run, "total_return_pct") else None,
                "total_trades": latest_run.total_trades if latest_run and hasattr(latest_run, "total_trades") else 0,
                "sharpe_ratio": float(latest_run.sharpe_ratio) if latest_run and hasattr(latest_run, "sharpe_ratio") else None,
            }
    except Exception as exc:
        logger.warning("Failed to load portfolio: %s", exc)
        return {"positions": [], "trades": [], "total_value": None, "total_return_pct": None, "total_trades": 0}


@app.get("/api/search")
def search_stocks(query: str = Query("", description="Search query"), limit: int = Query(20)):
    """Search stocks by symbol or company name."""
    results = search_tickers(query, limit=limit)
    return {
        "results": [
            {
                "sym": sym,
                "name": STOCK_INFO.get(sym, (sym, "USD"))[0],
                "currency": STOCK_INFO.get(sym, (sym, "USD"))[1],
            }
            for sym in results
        ]
    }


@app.get("/api/prices/stream")
async def stream_prices(symbols: str = Query(..., description="Comma-separated symbols")):
    """
    Server-Sent Events stream of live Finnhub trade prices.
    Subscribe to: GET /api/prices/stream?symbols=AAPL,MSFT,MC.PA
    Each event: data: {"symbol": "AAPL", "price": 182.5, "volume": 100, "ts": 1700000000000}
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    subscribe_symbols(sym_list)

    # Seed with latest known prices immediately
    snapshot = get_all_prices()

    q: queue.Queue = queue.Queue(maxsize=200)

    def on_trade(symbol: str, data: dict) -> None:
        if symbol in sym_list:
            try:
                q.put_nowait({"symbol": symbol, **data})
            except queue.Full:
                pass

    add_subscriber(on_trade)

    async def event_generator():
        try:
            # Send current snapshot first
            for sym in sym_list:
                if sym in snapshot:
                    d = snapshot[sym]
                    yield f"data: {json.dumps({'symbol': sym, **d})}\n\n"
            # Then stream live updates
            while True:
                try:
                    msg = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: q.get(timeout=25)
                    )
                    yield f"data: {json.dumps(msg)}\n\n"
                except Exception:
                    yield ": heartbeat\n\n"
        finally:
            remove_subscriber(on_trade)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════════
# TRADER V2 — Risk / Regime / Strategy stack
# ═══════════════════════════════════════════════════════════════════════════════════


@app.get("/api/v2/regime")
def get_regime(benchmark: str = Query("SPY")):
    """Current market regime signal from the benchmark index."""
    try:
        from app.regime.detector import RegimeDetector
        df = fetch_ohlcv(benchmark, interval="1d", period="2y")
        if df is None or df.empty:
            raise HTTPException(404, f"no data for {benchmark}")
        signal = RegimeDetector().detect(df["close"])
        return {"benchmark": benchmark, **signal.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v2/risk/status")
def get_risk_status():
    """Read-only snapshot of the risk manager configuration & breaker state."""
    try:
        from app.risk.manager import RiskManager
        rm = RiskManager()
        return rm.status()
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/v2/risk/kill-switch")
def activate_kill_switch(reason: str = Query("manual halt via API")):
    """Activate the emergency kill switch (halts all trading)."""
    from app.risk.kill_switch import KillSwitch
    ks = KillSwitch()
    ks.activate(reason)
    return {"active": ks.is_active, "reason": ks.reason}


@app.delete("/api/v2/risk/kill-switch")
def deactivate_kill_switch():
    """Deactivate the kill switch (resume trading)."""
    from app.risk.kill_switch import KillSwitch
    ks = KillSwitch()
    ks.deactivate()
    return {"active": ks.is_active}


# ═══════════════════════════════════════════════════════════════════════════════════
# AI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════


@app.get("/api/dashboard/charts")
def get_dashboard_charts():
    """Return weekly capital history + market indices history for dashboard charts."""
    from datetime import timedelta
    import yfinance as yf

    INDEX_META = [
        {"ticker": "^IXIC",  "name": "NASDAQ",   "color": "#5AFF8C"},
        {"ticker": "^DJI",   "name": "Dow Jones", "color": "#4DABF7"},
        {"ticker": "^FCHI",  "name": "CAC 40",    "color": "#FFD43B"},
        {"ticker": "^GDAXI", "name": "DAX 40",    "color": "#FF8787"},
    ]

    # ── Algo capital history (portfolio snapshots, last 7 days) ───────────────
    capital = {"dates": [], "values": [], "change_pct": 0.0}
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        with get_session() as session:
            snaps = (
                session.query(PortfolioModel)
                .filter(PortfolioModel.timestamp >= cutoff)
                .order_by(PortfolioModel.timestamp.asc())
                .all()
            )
        if snaps:
            dates = [s.timestamp[:10] if isinstance(s.timestamp, str) else str(s.timestamp)[:10] for s in snaps]
            values = [float(s.total_value) for s in snaps]
            first, last = values[0], values[-1]
            change_pct = round((last - first) / first * 100, 2) if first else 0.0
            capital = {"dates": dates, "values": values, "change_pct": change_pct}
    except Exception as exc:
        logger.warning("Capital history error: %s", exc)

    # ── ARIA capital history (last 7 days from aria_portfolio snapshots) ───────
    aria_capital = {"values": [], "change_pct": 0.0, "current": 0.0}
    try:
        from app.ai import aria_portfolio as _ap
        stats = _ap.get_stats()
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()
        recent = [s for s in stats["snapshots"] if s["ts"] >= cutoff_ts]
        if recent:
            vals = [round(s["value"], 2) for s in recent]
            first_v, last_v = vals[0], vals[-1]
            aria_capital = {
                "values": vals,
                "change_pct": round((last_v - first_v) / first_v * 100, 2) if first_v else 0.0,
                "current": last_v,
            }
    except Exception as exc:
        logger.debug("ARIA capital history error: %s", exc)

    # ── Market indices history (1h candles, 5d) ────────────────────────────────
    indices = []
    def _fetch_index(meta: dict) -> dict | None:
        try:
            ticker = yf.Ticker(meta["ticker"])
            hist = ticker.history(period="5d", interval="1h")
            if hist is None or hist.empty:
                return None
            close_col = "Close" if "Close" in hist.columns else "close"
            prices = [round(float(v), 2) for v in hist[close_col].tolist() if not pd.isna(v)]
            dates_raw = [str(idx) for idx in hist.index]
            if not prices:
                return None
            first = prices[0]
            last = prices[-1]
            change_pct = round((last - first) / first * 100, 2) if first else 0.0
            # Normalize to base 100 for multi-line comparison
            normalized = [round(v / first * 100, 3) for v in prices]
            return {
                "ticker": meta["ticker"],
                "name": meta["name"],
                "color": meta["color"],
                "dates": dates_raw,
                "values": prices,
                "normalized": normalized,
                "change_pct": change_pct,
                "last": last,
            }
        except Exception as exc:
            logger.warning("Index fetch failed %s: %s", meta["ticker"], exc)
            return None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_fetch_index, m): m for m in INDEX_META}
        for fut in as_completed(futs):
            result = fut.result()
            if result:
                indices.append(result)

    indices.sort(key=lambda x: INDEX_META.index(next(m for m in INDEX_META if m["ticker"] == x["ticker"])))

    return {"capital": capital, "aria_capital": aria_capital, "indices": indices}


@app.get("/api/ai/score/{symbol}")
def get_ai_score(symbol: str):
    """Return cached AI score + rationale + sources for a ticker."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        from app.ai import score_store
        cfg = _load_config()
        ttl = cfg.get("ai_analysis", {}).get("score_ttl_seconds", 3600)
        entry = score_store.get_entry(symbol.upper(), ttl=ttl)
        if entry is None:
            return {"available": False, "symbol": symbol.upper()}
        return {
            "available": True,
            "symbol": symbol.upper(),
            "score": entry["score"],
            "rationale": entry["rationale"],
            "sources": entry["sources"],
            "ts": entry["ts"],
            "age_seconds": int(time.time() - entry["ts"]),
        }
    except Exception as exc:
        logger.warning("AI score fetch error for %s: %s", symbol, exc)
        return {"available": False, "symbol": symbol.upper(), "error": str(exc)}


@app.post("/api/ai/score/{symbol}")
def trigger_ai_score(symbol: str):
    """Trigger an immediate on-demand AI analysis for a ticker (sync, uses requests)."""
    import requests as req
    import json as _json
    from datetime import datetime as _dt

    cfg = _load_config()
    ai_cfg = cfg.get("ai_analysis", {})
    api_key = _read_env_key("OPENROUTER_API_KEY")
    model = ai_cfg.get("model", "perplexity/sonar")
    timeout = ai_cfg.get("timeout_seconds", 30)

    if not api_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY non configurée")

    ticker = symbol.upper()
    prompt = (
        f"Ticker boursier : {ticker}\n"
        f"Date et heure : {_dt.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "Analyse le comportement probable de cette action sur les prochaines 24-48 heures "
        "en te basant sur les dernières actualités, la tendance du secteur, et le sentiment du marché.\n\n"
        "Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :\n"
        '{"score": <float 0.0-1.0>, "rationale": "<analyse concise en français, 2-3 phrases>", '
        '"sources": [{"title": "<titre>", "url": "<url complète>"}, ...]}\n\n'
        "0.0 = fort signal de vente, 0.5 = neutre, 1.0 = fort signal d'achat. "
        "Inclure 2 à 4 sources réelles et vérifiables."
    )
    try:
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
        content = r.json()["choices"][0]["message"]["content"]
        data = _json.loads(content)
        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
        rationale = data.get("rationale", "")
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        try:
            from app.ai import score_store
            score_store.set_score(ticker, score, rationale, sources)
        except Exception:
            pass

        return {
            "success": True, "available": True,
            "symbol": ticker, "score": score,
            "rationale": rationale, "sources": sources,
            "ts": time.time(), "age_seconds": 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("On-demand AI analysis failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ai/force")
def force_ai_analysis():
    """Force an immediate AI analysis cycle even if markets are closed."""
    try:
        from app.ai import scheduler as ai_scheduler
        ai_scheduler.force_now()
        return {"success": True, "message": "Analyse forcée déclenchée"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ai/models")
def get_ai_models():
    """Return all models available on OpenRouter."""
    import requests as req
    api_key = _read_env_key("OPENROUTER_API_KEY")
    headers = {"HTTP-Referer": "https://tradeflow.local", "X-Title": "TradeFlow"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = req.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        r.raise_for_status()
        models = r.json().get("data", [])
        return {"models": sorted(m["id"] for m in models if "id" in m)}
    except Exception as exc:
        logger.warning("Failed to fetch OpenRouter models: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenRouter unreachable: {exc}")


@app.post("/api/ai/test")
def test_ai_connection(body: dict = Body(default={})):
    """Test OpenRouter connection with the given API key and model."""
    import requests as req
    api_key = body.get("api_key") or _read_env_key("OPENROUTER_API_KEY")
    model = body.get("model", "perplexity/sonar")
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key provided")
    try:
        r = req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://tradeflow.local"},
            json={"model": model, "messages": [{"role": "user", "content": 'Reply with valid JSON only: {"score": 0.5}'}], "response_format": {"type": "json_object"}},
            timeout=15,
        )
        r.raise_for_status()
        return {"success": True, "model": model}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Autonomous AI agent ──────────────────────────────────────────────────────────

_AUTONOMOUS_PROMPT = """\
Tu es ARIA (Autonomous Research & Investment Agent), un agent de trading algorithmique expert \
avec plus de 20 ans d'expérience sur les marchés financiers mondiaux \
(NYSE, NASDAQ, Euronext, LSE, TSE).

TICKER ANALYSÉ : {ticker}
DATE ET HEURE  : {date}

SOURCES À CONSULTER OBLIGATOIREMENT :
1. Actualités financières récentes (Bloomberg, Reuters, FT, WSJ — dernières 48h)
2. Publications récentes sur X (Twitter/𝕏) : analystes, dirigeants, institutionnels, retail
3. Indicateurs techniques : tendance, momentum, niveaux support/résistance
4. Données fondamentales : valorisation, résultats récents, guidances, dividendes
5. Contexte sectoriel et macro-économique (taux, USD, matières premières)

RÈGLES DE GESTION DU RISQUE (OBLIGATOIRES) :
- confidence < 0.50 → action = "HOLD", position_size_pct = 0
- confidence 0.50–0.65 → position_size_pct ≤ 2 %, ratio R/R ≥ 1:2
- confidence 0.65–0.80 → position_size_pct ≤ 5 %, ratio R/R ≥ 1:2
- confidence > 0.80 → position_size_pct ≤ 10 %, ratio R/R ≥ 1:2.5
- take_profit_pct DOIT être ≥ 2 × stop_loss_pct

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ni après :
{{"action": "<BUY | SELL | HOLD>", "confidence": <float 0.0–1.0>,
  "position_size_pct": <float 0.0–10.0>, "stop_loss_pct": <float 0.5–15.0>,
  "take_profit_pct": <float 1.0–30.0>, "time_horizon": "<24h | 48h | 1 semaine>",
  "rationale": "<analyse détaillée en français 4–6 phrases, inclure données X si pertinentes>",
  "key_risks": "<2–3 risques principaux, concis>",
  "sources": [{{"title": "<titre>", "url": "<url>"}}]}}

Inclure 3 à 6 sources réelles et récentes (< 48h). Posts X acceptés.
"""


def _do_autonomous_analysis(ticker: str, cfg: dict) -> dict:
    """Run a single autonomous analysis using requests (sync). Returns decision dict."""
    import requests as req
    import json as _json
    from datetime import datetime as _dt

    ai_cfg = cfg.get("ai_analysis", {})
    api_key = _read_env_key("OPENROUTER_API_KEY")
    model = ai_cfg.get("model", "perplexity/sonar")
    timeout = ai_cfg.get("timeout_seconds", 45)

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY non configurée")

    prompt = _AUTONOMOUS_PROMPT.format(
        ticker=ticker.upper(),
        date=_dt.now().strftime("%Y-%m-%d %H:%M"),
    )

    r = req.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://tradeflow.local", "X-Title": "TradeFlow"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
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
    pos = max(0.0, min(10.0, float(data.get("position_size_pct", 0))))
    sl = max(0.5, min(15.0, float(data.get("stop_loss_pct", 2.0))))
    tp = max(1.0, min(30.0, float(data.get("take_profit_pct", 4.0))))
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    from app.ai import score_store
    score_store.set_decision(
        ticker.upper(), action, confidence, pos, sl, tp,
        data.get("time_horizon", "24h"),
        data.get("rationale", ""),
        data.get("key_risks", ""),
        sources,
    )

    # Execute against ARIA virtual portfolio if action is tradeable
    if action in ("BUY", "SELL"):
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="1d", interval="5m")
            if hist is not None and not hist.empty:
                current_price = float(hist.iloc[-1]["Close"])
                from app.ai import aria_portfolio as ap
                cfg2 = _load_config()
                ap.set_initial_capital(float(cfg2.get("aria_capital", 10000)))
                ap.execute_decision(ticker.upper(), action, pos, sl, tp, current_price)
        except Exception as exc:
            logger.warning("ARIA portfolio execution failed for %s: %s", ticker, exc)

    return score_store.get_decision(ticker.upper()) or {}


@app.get("/api/comparison")
def get_comparison():
    """Compare algo vs ARIA portfolio performance."""
    cfg = _load_config()

    # ── ARIA stats ────────────────────────────────────────────────────────────
    try:
        from app.ai import aria_portfolio as ap
        aria_stats = ap.get_stats()
    except Exception as exc:
        logger.warning("Could not load ARIA portfolio: %s", exc)
        aria_initial = cfg.get("aria_capital", 10000)
        aria_stats = {
            "initial_capital": aria_initial, "cash": aria_initial,
            "current_value": aria_initial, "pnl": 0, "pnl_pct": 0,
            "total_trades": 0, "open_positions": 0, "win_rate": 0,
            "positions": {}, "recent_trades": [], "snapshots": [],
        }

    # ── Algo stats from SQLite ────────────────────────────────────────────────
    algo_initial = cfg.get("algo_capital", cfg.get("default_capital", 10000))
    algo_stats: dict = {
        "initial_capital": algo_initial, "current_value": algo_initial,
        "pnl": 0, "pnl_pct": 0, "total_trades": 0, "win_rate": 0,
        "recent_trades": [], "snapshots": [],
    }
    try:
        with get_session() as session:
            run = (
                session.query(SimRun)
                .order_by(SimRun.id.desc())
                .first()
            )
            if run:
                snaps = (
                    session.query(Portfolio)
                    .filter_by(sim_run_id=run.id)
                    .order_by(Portfolio.timestamp)
                    .all()
                )
                history = [{"ts": s.timestamp.timestamp(), "value": s.total_value} for s in snaps]
                trades = session.query(Trade).filter_by(sim_run_id=run.id).order_by(Trade.timestamp.desc()).limit(20).all()
                cur = run.final_value or (history[-1]["value"] if history else run.initial_capital)
                pnl = cur - run.initial_capital
                algo_stats = {
                    "initial_capital": run.initial_capital,
                    "current_value": round(cur, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(run.total_return_pct or (pnl / run.initial_capital * 100 if run.initial_capital else 0), 2),
                    "total_trades": run.total_trades or 0,
                    "win_rate": round(run.win_rate or 0, 3),
                    "recent_trades": [t.to_dict() for t in trades],
                    "snapshots": history,
                }
    except Exception as exc:
        logger.warning("Could not load algo stats: %s", exc)

    return {"algo": algo_stats, "aria": aria_stats}


@app.post("/api/aria/reset")
def reset_aria_portfolio(body: dict = Body(default={})):
    """Reset ARIA virtual portfolio to its initial capital."""
    try:
        from app.ai import aria_portfolio as ap
        cfg = _load_config()
        capital = float(body.get("capital") or cfg.get("aria_capital", 10000))
        ap.reset(capital)
        return {"success": True, "message": f"Portfolio ARIA réinitialisé à {capital}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tickers")
def get_ticker_list():
    """Return the full list of all market tickers (no OHLCV fetch — instant)."""
    tickers = get_all_tickers()
    return {
        "tickers": [
            {"sym": s, "name": get_display_name(s)}
            for s in tickers
        ],
        "count": len(tickers),
    }


@app.get("/api/ai/decisions")
def get_all_ai_decisions():
    """Return all cached autonomous AI decisions."""
    from app.ai import score_store
    cfg = _load_config()
    ttl = cfg.get("ai_analysis", {}).get("score_ttl_seconds", 7200)
    decisions = score_store.get_all_decisions(ttl=ttl)
    return {"decisions": {k: {**v, "available": True, "symbol": k} for k, v in decisions.items()}}


@app.get("/api/ai/decisions/{symbol}")
def get_ai_decision(symbol: str):
    """Return cached autonomous decision for a ticker."""
    from app.ai import score_store
    cfg = _load_config()
    ttl = cfg.get("ai_analysis", {}).get("score_ttl_seconds", 7200)
    d = score_store.get_decision(symbol.upper(), ttl=ttl)
    if d is None:
        return {"available": False, "symbol": symbol.upper()}
    return {"available": True, "symbol": symbol.upper(), **d, "age_seconds": int(time.time() - d["ts"])}


@app.post("/api/ai/decisions/{symbol}")
def trigger_ai_decision(symbol: str):
    """Trigger an immediate autonomous analysis for a ticker."""
    try:
        cfg = _load_config()
        d = _do_autonomous_analysis(symbol.upper(), cfg)
        return {"available": True, "symbol": symbol.upper(), **d, "age_seconds": 0}
    except Exception as exc:
        logger.error("Autonomous analysis failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ai/analyze-all")
def ai_analyze_all():
    """Trigger autonomous analysis for ALL market tickers (background thread)."""
    import threading
    cfg = _load_config()
    tickers = get_all_tickers()

    def _run():
        for sym in tickers:
            try:
                _do_autonomous_analysis(sym, cfg)
                logger.info("Autonomous analysis done: %s", sym)
            except Exception as exc:
                logger.warning("Autonomous analysis failed for %s: %s", sym, exc)

    threading.Thread(target=_run, daemon=True, name="aria-analyze-all").start()
    return {"success": True, "message": f"Analyse ARIA lancée pour {len(tickers)} ticker(s)", "tickers": tickers}


# ═══════════════════════════════════════════════════════════════════════════════════
# STATIC FILES
# ═══════════════════════════════════════════════════════════════════════════════════


@app.get("/")
def serve_html():
    """Serve the main TradeFlow SPA."""
    if HTML_FILE.exists():
        return FileResponse(str(HTML_FILE))
    return {"error": "TradeFlow.html not found. Ensure it exists in the webui directory."}


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting TradeFlow server on 0.0.0.0:%s", port)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
