"""
TradeFlow — Modern Single-Page Trading Dashboard
All NASDAQ stocks, stream refresh, beginner-friendly French UI.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.analysis.composite import compute_composite_score
from app.bot.live_trader import create_live_session, get_active_live_session, stop_live_session
from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.data.nasdaq import get_all_tickers, get_display_name, get_currency, format_price, format_price_sign, search_tickers
from app.database.models import Portfolio as PortfolioModel, SimRun, Trade
from app.database.session import get_session, init_database
from app.strategies.base import Signal
from app.strategies.composite_strategy import CompositeStrategy
from app.webui.explanations import (
    card_class, explain_score, explain_sub_score, format_pnl,
    pnl_class, pnl_color, score_color, signal_badge_class, signal_label,
)

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

# Stream auto-refresh: 15s interval, silent (no page flash)
st_autorefresh(interval=15000, key="stream_refresh")

init_database()

# ── Load tickers ────────────────────────────────────────────────────────────────

ALL_TICKERS = get_all_tickers()
TICKER_DISPLAY = {sym: get_display_name(sym) for sym in ALL_TICKERS}

# Default selection
DEFAULT_PICKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "MC.PA"]


# ── SVG Gauge builder ──────────────────────────────────────────────────────────

def gauge_svg(score: float, size: int = 120) -> str:
    c = score_color(score)
    r = (size - 16) / 2
    cx, cy = size / 2, size / 2
    circumference = 2 * 3.14159 * r
    dashoffset = circumference * (1 - score)
    label = signal_label(score)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="tf-gauge">
        <circle class="tf-gauge-bg" cx="{cx}" cy="{cy}" r="{r}"
                stroke-dasharray="{circumference}" stroke-dashoffset="0"
                transform="rotate(-90 {cx} {cy})"/>
        <circle class="tf-gauge-fill" cx="{cx}" cy="{cy}" r="{r}"
                stroke="{c}" stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}"
                transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" dominant-baseline="middle"
              fill="{c}" font-size="22" font-weight="800" font-family="Inter,sans-serif">{score:.2f}</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" dominant-baseline="middle"
              fill="{c}" font-size="10" font-weight="600" font-family="Inter,sans-serif">{label}</text>
    </svg>"""


def sub_bar_html(label: str, value: float, explanation: str = "") -> str:
    c = score_color(value)
    pct = value * 100
    expl_text = f'<div class="tf-explain" style="font-size:0.7rem;margin-top:2px;">{explanation}</div>' if explanation else ""
    return f"""
    <div class="tf-sub-bar">
        <div class="tf-sub-bar-label">
            <span>{label}</span><span style="color:{c};font-weight:700;">{value:.2f}</span>
        </div>
        <div class="tf-sub-bar-track">
            <div class="tf-sub-bar-fill" style="width:{pct}%;background:{c};"></div>
        </div>
        {expl_text}
    </div>"""


def stock_card_html(symbol: str, price: float, score_data, show_detail: bool = False) -> str:
    s = score_data.combined
    cc = card_class(s)
    badge_cls = signal_badge_class(s)
    badge_label = signal_label(s)
    explain = explain_score(s)
    display_name = get_display_name(symbol)
    currency = get_currency(symbol)
    price_str = format_price(price, currency)
    detail_html = ""
    if show_detail:
        detail_html = f"""
        <div style="border-top:1px solid #1E2530;margin-top:12px;padding-top:12px;text-align:left;">
            {sub_bar_html("Graphiques", score_data.technical, explain_sub_score("Technique", score_data.technical))}
            {sub_bar_html("Sentiment", score_data.sentiment, explain_sub_score("Sentiment", score_data.sentiment))}
            {sub_bar_html("Volume", score_data.momentum, explain_sub_score("Momentum", score_data.momentum))}
            <div style="border-top:1px solid #1E2530;margin-top:8px;padding-top:8px;">
                {sub_bar_html("RSI", score_data.rsi_score)}
                {sub_bar_html("MACD", score_data.macd_score)}
                {sub_bar_html("Bollinger", score_data.bollinger_score)}
                {sub_bar_html("Tendance", score_data.sma_score)}
            </div>
            <div style="display:flex;gap:12px;justify-content:center;margin-top:8px;font-size:0.7rem;color:#8B949E;">
                <span>Fear & Greed: {score_data.fear_greed:.2f}</span>
                <span>News: {score_data.news_sentiment:.2f}</span>
            </div>
        </div>"""
    return f"""
    <div class="tf-card {cc}">
        <div class="tf-card-symbol">{display_name}</div>
        <div class="tf-card-price">{price_str}</div>
        {gauge_svg(s, size=120)}
        <div class="tf-badge {badge_cls}">{badge_label}</div>
        <div class="tf-explain">{explain}</div>
        {detail_html}
    </div>"""


def position_card_html(symbol: str, qty: float, avg: float, cur: float) -> str:
    unrealized = (cur - avg) * qty
    cls = pnl_class(unrealized)
    c = pnl_color(unrealized)
    display_name = get_display_name(symbol)
    currency = get_currency(symbol)
    pnl_str = format_price_sign(unrealized, currency)
    return f"""
    <div class="tf-position {cls}">
        <div>
            <div class="tf-position-symbol">{display_name}</div>
            <div class="tf-position-detail">{qty:.4f} actions &middot; {format_price(avg, currency)} &rarr; {format_price(cur, currency)}</div>
        </div>
        <div class="tf-position-pnl {cls}" style="color:{c};font-size:1rem;">{pnl_str}</div>
    </div>"""


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
    return f"""
    <div class="tf-trade-row">
        <span class="tf-trade-time">{time}</span>
        <span class="{side_cls}">{side_label}</span>
        <span class="tf-trade-symbol">{display_name}</span>
        <span class="tf-trade-qty">{qty}</span>
        <span class="tf-trade-price">{price}</span>
        {pnl_html}
        {reason_html}
    </div>"""


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="tf-header">
    <div style="display:flex;align-items:center;">
        <span class="tf-brand">TradeFlow</span>
        <span class="tf-brand-sub">Analyse automatique du marche</span>
    </div>
</div>
""", unsafe_allow_html=True)

active = get_active_live_session()

# ── Configuration ──────────────────────────────────────────────────────────────

with st.expander("Configuration", expanded=(active is None)):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        capital = st.number_input("Capital de depart ($)", min_value=500, max_value=10_000_000,
                                   value=10_000, step=500, key="cfg_capital")
    with c2:
        symbols = st.multiselect("Actions a surveiller", ALL_TICKERS,
                                   default=[s for s in DEFAULT_PICKS if s in ALL_TICKERS],
                                   format_func=lambda s: TICKER_DISPLAY.get(s, s),
                                   key="cfg_symbols")
        custom = st.text_input("Ajouter un ticker", placeholder="ex: COIN, ABNB, MC.PA", key="cfg_custom")
        if custom:
            for t in [x.strip().upper() for x in custom.split(",") if x.strip()]:
                if t not in symbols:
                    symbols.append(t)
    with c3:
        interval = st.selectbox("Frequence d'analyse", ["1h", "15m", "1d"],
                                 format_func=lambda x: {"1h": "Toutes les heures", "15m": "Tous les 15 min", "1d": "Une fois par jour"}.get(x, x),
                                 key="cfg_interval")
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        c_start, c_stop = st.columns(2)
        with c_start:
            if st.button("Demarrer le bot", use_container_width=True, type="primary",
                         disabled=(active is not None)):
                if not symbols:
                    st.error("Choisissez au moins une action.")
                else:
                    run_id = create_live_session("composite", symbols, interval, capital)
                    st.success(f"Session #{run_id} lancee !")
                    st.rerun()
        with c_stop:
            if st.button("Arreter le bot", use_container_width=True, disabled=(active is None)):
                stop_live_session()
                st.rerun()

st.markdown("---")

# ── No active session: show live preview ────────────────────────────────────────

if active is None:
    st.markdown('<div class="tf-section">Analyse en direct</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-sub">Apercu des scores — aucune session active</div>', unsafe_allow_html=True)

    preview_symbols = symbols if symbols else DEFAULT_PICKS[:4]
    cols = st.columns(min(len(preview_symbols), 4))

    for i, sym in enumerate(preview_symbols[:4]):
        with cols[i]:
            try:
                df = fetch_ohlcv(sym, interval=interval, period="3mo")
                if df is not None and not df.empty:
                    df = add_all_indicators(df)
                    df.attrs["symbol"] = sym
                    score = compute_composite_score(df, sym)
                    price = float(df.iloc[-1]["close"])
                    st.markdown(stock_card_html(sym, price, score), unsafe_allow_html=True)
            except Exception:
                st.warning(f"{sym}: erreur de chargement")

    st.markdown("---")
    st.markdown("""
    <div class="tf-empty">
        <div class="tf-empty-icon">1</div>
        <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">Comment ca marche ?</div>
        <div style="max-width:500px;margin:0.5rem auto;color:#8B949E;line-height:1.6;">
            <b>1.</b> Choisissez votre capital de depart (en euros)<br>
            <b>2.</b> Selectionnez les actions que le bot va surveiller<br>
            <b>3.</b> Cliquez <b>Demarrer le bot</b> — il analyse le marche et trade automatiquement<br><br>
            Le score va de <span style="color:#FF4B6E;">0 (vendre)</span> a
            <span style="color:#00C896;">1 (acheter)</span>. Quand le score depasse
            <b>0.70</b>, le bot achete. Quand il passe sous <b>0.30</b>, il vend.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Active session ────────────────────────────────────────────────────────────

run_id = active["id"]
symbols_live = [s.strip() for s in active["symbol"].split(",")]
initial_cap = active["initial_capital"]
last_tick = active.get("last_tick_at")

st.markdown(f"""
<div class="tf-status">
    <div class="tf-status-dot"></div>
    <span class="tf-status-text">Bot actif</span>
    <span class="tf-status-detail">Session #{run_id}</span>
    <span class="tf-status-detail">{', '.join(symbols_live)}</span>
    <span class="tf-status-detail">{active['interval']}</span>
    {"<span class='tf-status-detail'>Derniere analyse: " + (last_tick[:19].replace('T',' ') if last_tick else '—') + "</span>" if last_tick else ""}
</div>
""", unsafe_allow_html=True)

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
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;">
    <div class="tf-metric">
        <div class="tf-metric-value ${pnl_cls}" style="color:{pnl_c}">{total_value:,.2f} €</div>
        <div class="tf-metric-label">Valeur totale</div>
    </div>
    <div class="tf-metric">
        <div class="tf-metric-value ${pnl_cls}" style="color:{pnl_c}">{pnl_pct:+.2f}%</div>
        <div class="tf-metric-label">Rendement</div>
    </div>
    <div class="tf-metric">
        <div class="tf-metric-value">{cash:,.2f} €</div>
        <div class="tf-metric-label">Cash disponible</div>
    </div>
    <div class="tf-metric">
        <div class="tf-metric-value">{n_trades}</div>
        <div class="tf-metric-label">Trades executes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stock score cards ──────────────────────────────────────────────────────────

st.markdown('<div class="tf-section">Analyse par action</div>', unsafe_allow_html=True)
st.markdown('<div class="tf-section-sub">Score de 0 (vendre) a 1 (acheter) — le bot agit au-dessus de 0.70 et en dessous de 0.30</div>', unsafe_allow_html=True)

show_detail = st.checkbox("Voir les details", value=False, key="show_detail")

strategy = CompositeStrategy()

# Render cards in rows of 4
for row_start in range(0, len(symbols_live), 4):
    row_syms = symbols_live[row_start:row_start + 4]
    card_cols = st.columns(len(row_syms))
    for i, symbol in enumerate(row_syms):
        with card_cols[i]:
            try:
                df = fetch_ohlcv(symbol, interval=active["interval"], period="3mo")
                if df is not None and not df.empty:
                    df = add_all_indicators(df)
                    df.attrs["symbol"] = symbol
                    score = compute_composite_score(df, symbol)
                    price = float(df.iloc[-1]["close"])
                    st.markdown(stock_card_html(symbol, price, score, show_detail=show_detail),
                                unsafe_allow_html=True)
            except Exception:
                st.warning(f"{symbol}: erreur")

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
        st.markdown("""
        <div class="tf-empty">
            <div>En attente de la premiere analyse du bot...</div>
        </div>
        """, unsafe_allow_html=True)

with col_pos:
    st.markdown('<div class="tf-section">Positions ouvertes</div>', unsafe_allow_html=True)
    if positions:
        for sym, pos in positions.items():
            qty = pos.get("qty", 0)
            avg = pos.get("avg_price", 0)
            cur = pos.get("current_price", avg)
            st.markdown(position_card_html(sym, qty, avg, cur), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tf-empty">
            <div>Aucune position ouverte</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Trade history ──────────────────────────────────────────────────────────────

st.markdown('<div class="tf-section">Historique des trades</div>', unsafe_allow_html=True)

if trades_df.empty:
    st.markdown("""
    <div class="tf-empty">
        <div>Aucun trade execute pour l'instant</div>
        <div class="tf-explain">Le bot attend que le score depasse 0.70 pour acheter ou passe sous 0.30 pour vendre</div>
    </div>
    """, unsafe_allow_html=True)
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

    st.markdown(f"""
    <div style="border:1px solid #1E2530;border-radius:12px;overflow:hidden;">
        {html_rows}
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="tf-footer">
    Flux automatique &middot; Derniere mise a jour: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)