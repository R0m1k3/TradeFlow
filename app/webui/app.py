"""
TradeFlow — Modern Single-Page Trading Dashboard
All NASDAQ + European stocks, stream refresh, beginner-friendly French UI.
Config via gear icon modal. Every ticker gets a card.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio
import json
import logging
import signal
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from app.ai import scheduler as ai_scheduler
from app.ai import score_store
from app.ai.openrouter_client import fetch_models, test_connection
from app.ai.persist import get_all_latest_ai_signals
from app.analysis.composite import compute_composite_score
from app.bot.live_trader import get_active_live_session, stop_live_session, resume_or_create_live_session
from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.data.nasdaq import get_all_tickers, get_display_name, get_currency, format_price, format_price_sign, search_tickers
from app.data.markets import get_all_market_statuses, any_market_open, next_market_event, EXCHANGES
from app.database.models import Portfolio as PortfolioModel, SimRun, Trade
from app.database.session import get_session, init_database
from app.strategies.base import Signal
from app.strategies.composite_strategy import CompositeStrategy
from app.webui.explanations import (
    pnl_class, pnl_color, score_color, signal_badge_class, signal_label,
)

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradeFlow",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject CSS
_css_path = Path(__file__).parent / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}</style>", unsafe_allow_html=True)

# Stream auto-refresh: 15s interval, silent
st_autorefresh(interval=60000, key="stream_refresh")

if "db_initialized" not in st.session_state:
    init_database()
    st.session_state.db_initialized = True

# ── Config helpers ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)


def _read_env_key(name: str) -> str:
    if not _ENV_PATH.exists():
        return ""
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch_models(api_key: str) -> list[str]:
    return fetch_models(api_key=api_key)


def _write_env_key(name: str, value: str) -> None:
    lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.exists() else []
    lines = [l for l in lines if not l.startswith(f"{name}=")]
    if value:
        lines.append(f"{name}={value}")
    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Bot process management ───────────────────────────────────────────────────────
# The bot runs as a separate subprocess. We track it via a PID file.

BOT_PID_FILE = Path(__file__).resolve().parents[2] / "data" / "bot.pid"


def _start_bot_process() -> None:
    """Launch run_bot.py as a background subprocess and save its PID."""
    if _get_bot_pid() is not None:
        logger = logging.getLogger(__name__)
        logger.info("Bot already running — skipping duplicate start")
        return

    bot_script = Path(__file__).resolve().parents[2] / "app" / "bot" / "run_bot.py"
    log_dir = BOT_PID_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "bot.log"
    proc = subprocess.Popen(
        [sys.executable, str(bot_script)],
        stdout=open(log_file, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    BOT_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    BOT_PID_FILE.write_text(str(proc.pid), encoding="utf-8")


def _stop_bot_process() -> None:
    """Kill the bot subprocess and remove the PID file."""
    pid = _get_bot_pid()
    if pid is not None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            pass
        BOT_PID_FILE.unlink(missing_ok=True)


def _get_bot_pid() -> int | None:
    """Read the bot PID from the PID file. Returns None if missing or stale."""
    if not BOT_PID_FILE.exists():
        return None
    try:
        pid = int(BOT_PID_FILE.read_text(encoding="utf-8").strip())
        # Verify the process is still alive
        if sys.platform == "win32":
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                    capture_output=True, text=True)
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


# Clean up stale PID on startup
if BOT_PID_FILE.exists():
    if _get_bot_pid() is None:
        BOT_PID_FILE.unlink(missing_ok=True)

# Auto-restart bot if DB says running but process is gone
active_session = get_active_live_session()
if active_session is not None and _get_bot_pid() is None:
    _start_bot_process()
    logger.info("Auto-restarted bot for live session #%d", active_session["id"])


# ── Load tickers ────────────────────────────────────────────────────────────────

ALL_TICKERS = get_all_tickers()
TICKER_DISPLAY = {sym: get_display_name(sym) for sym in ALL_TICKERS}

# Start AI scheduler once tickers are known
_ai_cfg = _load_config().get("ai_analysis", {})
if _ai_cfg.get("enabled", False) and _read_env_key("OPENROUTER_API_KEY"):
    ai_scheduler.start(ALL_TICKERS)


# ── Stock table builder ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ticker_row(sym: str, interval: str, period: str) -> dict | None:
    """Fetch OHLCV + compute composite score for one ticker. Cached 5 min."""
    try:
        df = fetch_ohlcv(sym, interval=interval, period=period)
        if df is None or df.empty:
            return None
        df = add_all_indicators(df)
        df.attrs["symbol"] = sym
        score_data = compute_composite_score(df, sym)
        price = float(df.iloc[-1]["close"])
        prev_close = float(df.iloc[-2]["close"]) if len(df) >= 2 else price
        return {
            "price": price,
            "change_pct": ((price - prev_close) / prev_close * 100) if prev_close else 0.0,
            "score": score_data.combined,
        }
    except Exception:
        return None


def stock_table_html(symbols: list[str], interval: str, period: str = "3mo") -> str:
    """Generate a self-contained HTML table of stock data (parallel fetch, cached per ticker)."""
    import concurrent.futures

    ai_cfg = _load_config().get("ai_analysis", {})
    show_ai = ai_cfg.get("enabled", False)

    # Load persisted AI signals from DB (one query for all symbols)
    ai_signals = get_all_latest_ai_signals(interval=interval) if show_ai else {}

    # Parallel fetch — up to 8 tickers at a time
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_ticker_row, sym, interval, period): sym for sym in symbols}
        results = {sym: fut.result() for fut, sym in [(f, futures[f]) for f in concurrent.futures.as_completed(futures)]}

    rows_html = ""
    for sym in symbols:
        data = results.get(sym)
        if data is None:
            continue
        price = data["price"]
        change_pct = data["change_pct"]
        score = data["score"]

        display_name = get_display_name(sym)
        currency = get_currency(sym)
        price_str = format_price(price, currency)

        if change_pct > 0:
            arrow = "&#9650;"
            change_color = "#00C896"
        elif change_pct < 0:
            arrow = "&#9660;"
            change_color = "#FF4B6E"
        else:
            arrow = "&#9646;"
            change_color = "#8B949E"

        badge_cls = signal_badge_class(score)
        badge_label = signal_label(score)

        ai_cell = ""
        if show_ai:
            ai_data = ai_signals.get(sym)
            if ai_data:
                ai_action = ai_data.get("action", "N/A")
                ai_conf = ai_data.get("confidence")
                ai_score = ai_data.get("score")
                if ai_action:
                    if ai_action == "BUY":
                        ai_color = "#00C896"
                    elif ai_action == "SELL":
                        ai_color = "#FF4B6E"
                    else:
                        ai_color = "#8B949E"
                    conf_str = f" ({ai_conf:.0%})" if ai_conf is not None else ""
                    ai_cell = f'<td class="tf-td-score" style="color:{ai_color};font-weight:700;font-size:0.8rem;">{ai_action}{conf_str}</td>'
                elif ai_score is not None:
                    ai_cell = f'<td class="tf-td-score" style="color:{score_color(ai_score)};">{ai_score:.2f}</td>'
                else:
                    ai_cell = '<td class="tf-td-score" style="color:#8B949E;">—</td>'
            else:
                ai_cell = '<td class="tf-td-score" style="color:#8B949E;">—</td>'

        rows_html += (
            f'<tr>'
            f'<td class="tf-td-sym">{sym}</td>'
            f'<td class="tf-td-name">{display_name}</td>'
            f'<td class="tf-td-price">{price_str}</td>'
            f'<td class="tf-td-change" style="color:{change_color};">{arrow} {change_pct:+.2f}%</td>'
            f'<td class="tf-td-score" style="color:{score_color(score)};">{score:.2f}</td>'
            f'{ai_cell}'
            f'<td class="tf-td-signal"><span class="tf-badge {badge_cls}">{badge_label}</span></td>'
            f'</tr>'
        )

    if not rows_html:
        return '<div class="tf-empty">Aucune donnee disponible</div>'

    ai_header = '<th>Signal IA</th>' if show_ai else ''
    return (
        f'<div class="tf-table-wrapper">'
        f'<table class="tf-stock-table">'
        f'<thead><tr>'
        f'<th>Symbole</th><th>Societe</th><th>Prix</th>'
        f'<th>Variation</th><th>Score</th>{ai_header}<th>Signal</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


def position_card_html(symbol: str, qty: float, avg: float, cur: float) -> str:
    unrealized = (cur - avg) * qty
    cls = pnl_class(unrealized)
    c = pnl_color(unrealized)
    display_name = get_display_name(symbol)
    currency = get_currency(symbol)
    pnl_str = format_price_sign(unrealized, currency)
    return (
        f'<div class="tf-position {cls}">'
        f'<div>'
        f'<div class="tf-position-symbol">{display_name}</div>'
        f'<div class="tf-position-detail">{qty:.4f} actions &middot; {format_price(avg, currency)} &rarr; {format_price(cur, currency)}</div>'
        f'</div>'
        f'<div class="tf-position-pnl {cls}" style="color:{c};font-size:1rem;">{pnl_str}</div>'
        f'</div>'
    )


def trade_row_html(time: str, side: str, symbol: str, qty: str, price: str, pnl: str = "", reason: str = "", currency: str = "EUR") -> str:
    side_cls = "tf-trade-side-buy" if side == "BUY" else "tf-trade-side-sell"
    side_label = "ACHAT" if side == "BUY" else "VENTE"
    display_name = get_display_name(symbol)
    pnl_html = ""
    if pnl:
        val = float(pnl.replace("€", "").replace("$", "").replace(",", "").replace("+", ""))
        c = pnl_color(val)
        pnl_html = f'<span class="tf-trade-pnl {pnl_class(val)}" style="color:{c}">{pnl}</span>'
    reason_html = f'<span style="color:#8B949E;font-size:0.75rem;margin-left:auto;">{reason}</span>' if reason else ""
    return (
        f'<div class="tf-trade-row">'
        f'<span class="tf-trade-time">{time}</span>'
        f'<span class="{side_cls}">{side_label}</span>'
        f'<span class="tf-trade-symbol">{display_name}</span>'
        f'<span class="tf-trade-qty">{qty}</span>'
        f'<span class="tf-trade-price">{price}</span>'
        f'{pnl_html}'
        f'{reason_html}'
        f'</div>'
    )


# ── Header with gear icon ──────────────────────────────────────────────────────

active = get_active_live_session()

st.markdown(
    '<div class="tf-header">'
    '<div style="display:flex;align-items:center;">'
    '<span class="tf-brand">TradeFlow</span>'
    '<span class="tf-brand-sub">Analyse automatique du marche</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Gear icon button for config modal
col_title, col_gear = st.columns([10, 1])
with col_gear:
    gear_clicked = st.button("⚙️", key="gear_btn", help="Configuration")

# ── Market overview cards ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_market_statuses() -> list[dict]:
    return get_all_market_statuses()


def market_card_html(status: dict) -> str:
    is_open = status["open"]
    state_cls = "open" if is_open else "closed"
    badge_cls = "tf-market-badge open" if is_open else "tf-market-badge closed"
    badge_label = "OUVERT" if is_open else "FERME"
    price = status.get("price")
    price_str = f"{price:,.2f} pts" if price is not None else "—"

    # Price direction indicator
    change_html = ""
    if price is not None and status.get("prev_close") is not None and status["prev_close"] > 0:
        prev = status["prev_close"]
        change_pct = ((price - prev) / prev) * 100
        if change_pct > 0:
            arrow = "&#9650;"
            chg_color = "#00C896"
        elif change_pct < 0:
            arrow = "&#9660;"
            chg_color = "#FF4B6E"
        else:
            arrow = "&#9646;"
            chg_color = "#8B949E"
        change_html = (
            f'<div class="tf-market-change" '
            f'style="color:{chg_color};font-weight:700;font-size:0.85rem;margin-top:2px;">'
            f'{arrow} {change_pct:+.2f}%</div>'
        )

    return (
        f'<div class="tf-market-card {state_cls}">'
        f'<div class="tf-market-name">{status["name"]}</div>'
        f'<div class="tf-market-index">{status["index_name"]}</div>'
        f'<div class="tf-market-price">{price_str}</div>'
        f'{change_html}'
        f'<div class="{badge_cls}">{badge_label}</div>'
        f'</div>'
    )


market_statuses = load_market_statuses()
markets_html = "".join(market_card_html(m) for m in market_statuses)
any_open = any(m["open"] for m in market_statuses)

# Market status summary
if any_open:
    summary_color = "#00C896"
    summary_text = "Marches ouverts — le bot peut trader"
else:
    event_type, event_time = next_market_event()
    from datetime import datetime as _dt, timezone
    now = _dt.now(timezone.utc)
    wait_min = int(max(0, (event_time - now).total_seconds()) // 60)
    summary_color = "#8B949E"
    summary_text = f"Marches fermes — ouverture dans ~{wait_min} min"

pulse_anim = "animation:tf-pulse 2s infinite;" if any_open else ""

# AI last-check badge
_ai_cfg_now = _load_config().get("ai_analysis", {})
_ai_enabled_now = _ai_cfg_now.get("enabled", False)
_ai_status_now = ai_scheduler.get_status()
_ai_last_now = _ai_status_now.get("last_run_at")
if _ai_enabled_now:
    if _ai_last_now:
        from datetime import datetime as _dt2, timezone as _tz2
        _elapsed_now = int((_dt2.now(_tz2.utc).timestamp() - _ai_last_now) / 60)
        _ai_dt_now = _dt2.fromtimestamp(_ai_last_now, tz=_tz2.utc).strftime("%H:%M")
        _mode_now = {"autonomous": "ARIA", "hybrid": "Hybride"}.get(_ai_status_now.get("last_run_mode", ""), "IA")
        _tickers_now = _ai_status_now.get("last_run_ticker_count", 0)
        ai_badge_html = (
            f'<span style="color:#A78BFA;font-size:0.82rem;font-weight:600;">'
            f'🧠 {_mode_now} — derniere verification: {_ai_dt_now} '
            f'<span style="color:#8B949E;">({_elapsed_now} min · {_tickers_now} tickers)</span>'
            f'</span>'
        )
    else:
        ai_badge_html = '<span style="color:#8B949E;font-size:0.82rem;">🧠 IA activee — en attente du 1er cycle...</span>'
else:
    ai_badge_html = '<span style="color:#8B949E;font-size:0.82rem;">🧠 IA desactivee</span>'

st.markdown(
    f'<div class="tf-market-grid">{markets_html}</div>'
    f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:1rem;flex-wrap:wrap;">'
    f'<div style="display:flex;align-items:center;gap:8px;">'
    f'<div style="width:8px;height:8px;background:{summary_color};border-radius:50%;{pulse_anim}"></div>'
    f'<span style="color:{summary_color};font-size:0.85rem;font-weight:600;">{summary_text}</span>'
    f'</div>'
    f'<span style="color:#30363D;">|</span>'
    f'{ai_badge_html}'
    f'</div>'
    f'<style>@keyframes tf-pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>',
    unsafe_allow_html=True,
)

# ── Config modal (opened by gear icon) ─────────────────────────────────────────

if gear_clicked:
    @st.dialog("Configuration", width="large")
    def config_modal():
        tab_bot, tab_ai = st.tabs(["🤖 Bot de trading", "🧠 Analyse IA"])

        # ── Tab Bot ────────────────────────────────────────────────────────────
        with tab_bot:
            st.markdown("### Parametres du bot")
            active_in_dialog = get_active_live_session()

            c1, c2 = st.columns(2)
            with c1:
                capital = st.number_input("Capital de depart (€)", min_value=500, max_value=10_000_000,
                                           value=10_000, step=500, key="dlg_capital")
            with c2:
                interval = st.selectbox("Frequence d'analyse", ["1h", "15m", "1d"],
                                         format_func=lambda x: {"1h": "Toutes les heures", "15m": "Tous les 15 min", "1d": "Une fois par jour"}.get(x, x),
                                         key="dlg_interval")

            st.info(f"Le bot analysera les **{len(ALL_TICKERS)} actions** disponibles (NASDAQ + Europe).")
            st.markdown(f"<div style='color:#8B949E;font-size:0.8rem;'>{len(ALL_TICKERS)} tickers : {', '.join(ALL_TICKERS[:12])}...</div>", unsafe_allow_html=True)

            c_start, c_stop = st.columns(2)
            with c_start:
                if st.button("▶ Demarrer le bot", use_container_width=True, type="primary",
                             disabled=(active_in_dialog is not None), key="dlg_start"):
                    run_id, resumed = resume_or_create_live_session("composite", ALL_TICKERS, interval, capital)
                    _start_bot_process()
                    msg = f"Session #{run_id} reprise" if resumed else f"Session #{run_id} lancee avec {len(ALL_TICKERS)} actions !"
                    st.success(msg)
                    st.rerun()
            with c_stop:
                if st.button("■ Arreter le bot", use_container_width=True,
                             disabled=(active_in_dialog is None), key="dlg_stop"):
                    _stop_bot_process()
                    stop_live_session()
                    st.rerun()

        # ── Tab IA ─────────────────────────────────────────────────────────────
        with tab_ai:
            st.markdown("### Analyse IA via OpenRouter")
            st.markdown(
                "<div style='color:#8B949E;font-size:0.85rem;margin-bottom:1rem;'>"
                "Un modele IA analysera chaque action toutes les 30 minutes et attribuera "
                "un score 0-1 pour affiner les decisions du bot."
                "</div>",
                unsafe_allow_html=True,
            )

            cfg = _load_config()
            ai_cfg = cfg.setdefault("ai_analysis", {})

            # Current saved key (masked display)
            saved_key = _read_env_key("OPENROUTER_API_KEY")
            key_hint = f"Cle actuelle : {saved_key[:8]}••••" if saved_key else "Aucune cle configuree"
            st.caption(key_hint)

            new_key = st.text_input(
                "Cle API OpenRouter",
                value="",
                type="password",
                placeholder="sk-or-v1-...",
                key="dlg_or_key",
                help="Votre cle API sur openrouter.ai — jamais stockee en clair dans le code",
            )

            current_model = ai_cfg.get("model", "perplexity/sonar")
            with st.spinner("Chargement des modeles OpenRouter..."):
                available_models = _cached_fetch_models(api_key=saved_key or new_key)
            if not available_models:
                st.warning("Impossible de recuperer la liste des modeles. Verifiez votre connexion ou entrez votre cle API.")
                available_models = [current_model] if current_model else ["perplexity/sonar"]
            model_idx = available_models.index(current_model) if current_model in available_models else 0
            selected_model = st.selectbox(
                f"Modele IA ({len(available_models)} disponibles)",
                available_models,
                index=model_idx,
                key="dlg_or_model",
                help="perplexity/sonar(-pro) recommande : acces web natif pour les actualites en temps reel",
            )

            ai_enabled = st.toggle(
                "Activer l'analyse IA",
                value=ai_cfg.get("enabled", False),
                key="dlg_ai_enabled",
            )

            score_weight = st.slider(
                "Poids du score IA dans la decision",
                min_value=0.0, max_value=1.0,
                value=float(ai_cfg.get("score_weight", 0.3)),
                step=0.05,
                key="dlg_score_weight",
                help="0 = l'IA n'influence pas le bot | 1 = decision basee uniquement sur l'IA",
            )

            col_save, col_test = st.columns(2)

            with col_save:
                if st.button("💾 Sauvegarder", use_container_width=True, type="primary", key="dlg_ai_save"):
                    if new_key:
                        _write_env_key("OPENROUTER_API_KEY", new_key)
                    ai_cfg["model"] = selected_model
                    ai_cfg["enabled"] = ai_enabled
                    ai_cfg["score_weight"] = score_weight
                    _save_config(cfg)
                    if ai_enabled and (new_key or saved_key):
                        ai_scheduler.start(ALL_TICKERS)
                    elif not ai_enabled:
                        ai_scheduler.stop()
                    st.success("Configuration IA sauvegardee !")

            with col_test:
                if st.button("🔌 Tester la connexion", use_container_width=True, key="dlg_ai_test"):
                    key_to_test = new_key or saved_key
                    if not key_to_test:
                        st.error("Entrez d'abord une cle API.")
                    else:
                        with st.spinner("Connexion a OpenRouter..."):
                            ok = asyncio.run(test_connection(key_to_test, selected_model))
                        if ok:
                            st.success(f"Connexion reussie — modele : {selected_model}")
                        else:
                            st.error("Echec — verifiez votre cle sur openrouter.ai")

    config_modal()

st.markdown("---")

# ── No active session: show all tickers in table ────────────────────────────────

if active is None:
    st.markdown('<div class="tf-section">Analyse en direct</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-sub">Apercu des scores — cliquez ⚙️ pour configurer et demarrer le bot</div>', unsafe_allow_html=True)

    # Search/filter
    search_query = st.text_input("Rechercher une action", placeholder="ex: Apple, NVDA, LVMH...", key="ticker_search")

    # Filter tickers based on search
    if search_query:
        filtered = search_tickers(search_query, limit=len(ALL_TICKERS))
    else:
        filtered = ALL_TICKERS

    # Render table
    if filtered:
        with st.spinner("Analyse des marches en cours..."):
            st.markdown(stock_table_html(filtered, interval="1h", period="3mo"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="tf-empty">'
        '<div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">Comment ca marche ?</div>'
        '<div style="max-width:500px;margin:0.5rem auto;color:#8B949E;line-height:1.6;">'
        '<b>1.</b> Cliquez sur ⚙️ en haut a droite pour configurer le capital<br>'
        '<b>2.</b> Le bot analyse automatiquement toutes les actions disponibles<br>'
        '<b>3.</b> Cliquez <b>Demarrer le bot</b> — il trade automatiquement<br><br>'
        'Le score va de <span style="color:#FF4B6E;">0 (vendre)</span> a '
        '<span style="color:#00C896;">1 (acheter)</span>. Quand le score depasse '
        '<b>0.70</b>, le bot achete. Quand il passe sous <b>0.30</b>, il vend.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Active session ────────────────────────────────────────────────────────────

run_id = active["id"]
symbols_live = [s.strip() for s in active["symbol"].split(",")]
initial_cap = active["initial_capital"]
last_tick = active.get("last_tick_at")

last_tick_html = f"<span class='tf-status-detail'>Derniere analyse algo: {last_tick[:19].replace('T',' ')}</span>" if last_tick else ""

# AI last check
_ai_status = ai_scheduler.get_status()
_ai_last = _ai_status.get("last_run_at")
if _ai_last:
    from datetime import datetime as _dt, timezone as _tz
    _ai_dt = _dt.fromtimestamp(_ai_last, tz=_tz.utc).strftime("%Y-%m-%d %H:%M")
    _ai_elapsed = int((_dt.now(_tz.utc).timestamp() - _ai_last) / 60)
    _ai_mode_label = {"autonomous": "ARIA", "hybrid": "Hybride"}.get(_ai_status.get("last_run_mode", ""), "IA")
    ai_last_html = (
        f"<span class='tf-status-detail' style='color:#A78BFA;'>"
        f"Derniere verif {_ai_mode_label}: {_ai_dt} ({_ai_elapsed} min)"
        f"</span>"
    )
else:
    ai_last_html = "<span class='tf-status-detail' style='color:#8B949E;'>IA: aucune analyse</span>"

st.markdown(
    f'<div class="tf-status">'
    f'<div class="tf-status-dot"></div>'
    f'<span class="tf-status-text">Bot actif</span>'
    f'<span class="tf-status-detail">Session #{run_id}</span>'
    f'<span class="tf-status-detail">{len(symbols_live)} actions</span>'
    f'<span class="tf-status-detail">{active["interval"]}</span>'
    f'{last_tick_html}'
    f'{ai_last_html}'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_recent_trades(run_id: int) -> pd.DataFrame:
    session = get_session()
    try:
        trades = session.query(Trade).filter_by(sim_run_id=run_id).order_by(Trade.timestamp.desc()).limit(50).all()
        return pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_equity_snapshots(run_id: int) -> pd.DataFrame:
    session = get_session()
    try:
        snaps = session.query(PortfolioModel).filter_by(sim_run_id=run_id).order_by(PortfolioModel.timestamp.asc()).all()
        if not snaps:
            return pd.DataFrame()
        return pd.DataFrame([{"timestamp": s.timestamp, "total_value": s.total_value, "cash": s.cash} for s in snaps])
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_latest_positions(run_id: int) -> dict:
    session = get_session()
    try:
        snap = session.query(PortfolioModel).filter_by(sim_run_id=run_id).order_by(PortfolioModel.timestamp.desc()).first()
        if snap:
            return {"cash": snap.cash, "total_value": snap.total_value, "positions": json.loads(snap.positions_json)}
        return {}
    finally:
        session.close()


trades_df = load_recent_trades(run_id)
equity_df = load_equity_snapshots(run_id)
latest = load_latest_positions(run_id)

cash = latest.get("cash", initial_cap)
total_value = latest.get("total_value", initial_cap)
positions = latest.get("positions", {})
pnl = total_value - initial_cap
pnl_pct = pnl / initial_cap * 100
n_trades = len(trades_df) if not trades_df.empty else 0


# ── Big metrics row ────────────────────────────────────────────────────────────

pnl_c = pnl_color(pnl)
pnl_cls = pnl_class(pnl)
st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;">'
    f'<div class="tf-metric"><div class="tf-metric-value" style="color:{pnl_c}">{total_value:,.2f} €</div><div class="tf-metric-label">Valeur totale</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value" style="color:{pnl_c}">{pnl_pct:+.2f}%</div><div class="tf-metric-label">Rendement</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value">{cash:,.2f} €</div><div class="tf-metric-label">Cash disponible</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value">{n_trades}</div><div class="tf-metric-label">Trades executes</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Stock score table — ALL tickers ──────────────────────────────────────────────

st.markdown('<div class="tf-section">Analyse par action</div>', unsafe_allow_html=True)
st.markdown('<div class="tf-section-sub">Score de 0 (vendre) a 1 (acheter) — le bot agit au-dessus de 0.70 et en dessous de 0.30</div>', unsafe_allow_html=True)

live_search = st.text_input("Rechercher", placeholder="ex: Apple, NVDA, LVMH...", key="live_ticker_search")

# Filter live symbols
if live_search:
    display_syms = [s for s in symbols_live if live_search.upper() in s or live_search.lower() in get_display_name(s).lower()]
else:
    display_syms = symbols_live

# Render table
if display_syms:
    st.markdown(stock_table_html(display_syms, interval=active["interval"], period="3mo"), unsafe_allow_html=True)

st.markdown("---")

# ── Equity curve + positions ───────────────────────────────────────────────────

col_chart, col_pos = st.columns([3, 1])

with col_chart:
    st.markdown('<div class="tf-section">Evolution du portefeuille</div>', unsafe_allow_html=True)
    if not equity_df.empty:
        color = "#00C896" if total_value >= initial_cap else "#FF4B6E"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["timestamp"], y=equity_df["total_value"],
            mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(0,200,150,0.06)" if color == "#00C896" else "rgba(255,75,110,0.06)",
            name="Valeur",
        ))
        fig.add_hline(y=initial_cap, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                       annotation_text=f"Capital {initial_cap:,.0f} €",
                       annotation_font_color="#8B949E", annotation_font_size=11)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111418", plot_bgcolor="#111418",
            height=300, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
            font=dict(family="Inter,sans-serif", size=12, color="#8B949E"),
        )
        fig.update_xaxes(gridcolor="#1E2530", color="#8B949E")
        fig.update_yaxes(gridcolor="#1E2530", color="#8B949E")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="tf-empty"><div>En attente de la premiere analyse du bot...</div></div>', unsafe_allow_html=True)

with col_pos:
    st.markdown('<div class="tf-section">Positions ouvertes</div>', unsafe_allow_html=True)
    if positions:
        for sym, pos in positions.items():
            qty = pos.get("qty", 0)
            avg = pos.get("avg_price", 0)
            cur = pos.get("current_price", avg)
            st.markdown(position_card_html(sym, qty, avg, cur), unsafe_allow_html=True)
    else:
        st.markdown('<div class="tf-empty"><div>Aucune position ouverte</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Trade history ──────────────────────────────────────────────────────────────

st.markdown('<div class="tf-section">Historique des trades</div>', unsafe_allow_html=True)

if trades_df.empty:
    st.markdown(
        '<div class="tf-empty">'
        '<div>Aucun trade execute pour l\'instant</div>'
        '<div class="tf-explain">Le bot attend que le score depasse 0.70 pour acheter ou passe sous 0.30 pour vendre</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    html_rows = ""
    for _, row in trades_df.head(20).iterrows():
        t = pd.to_datetime(row.get("timestamp", "")).strftime("%H:%M") if pd.notna(row.get("timestamp")) else ""
        side = row.get("side", "")
        sym = row.get("symbol", "")
        qty = f"{row.get('quantity', 0):.4f}" if pd.notna(row.get("quantity")) else ""
        price_val = row.get("price", 0)
        currency = get_currency(sym)
        price = format_price(price_val, currency) if pd.notna(price_val) else ""
        pnl_val = row.get("pnl", 0)
        pnl_str = format_price_sign(pnl_val, currency) if pd.notna(pnl_val) and side == "SELL" else ""
        reason = row.get("reason", "")
        if reason and len(reason) > 60:
            reason = reason[:57] + "..."
        html_rows += trade_row_html(t, side, sym, qty, price, pnl_str, reason, currency)

    st.markdown(
        f'<div style="border:1px solid #1E2530;border-radius:12px;overflow:hidden;">{html_rows}</div>',
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="tf-footer">Flux automatique &middot; Derniere mise a jour: {datetime.now().strftime("%H:%M:%S")}</div>',
    unsafe_allow_html=True,
)