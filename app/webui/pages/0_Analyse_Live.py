"""
TradeFlow — Live Trading Dashboard
Control the real-time bot and monitor its activity:
  • Start / Stop the live session
  • Current positions and portfolio value
  • Recent trades with reasons
  • Signal the bot would fire RIGHT NOW per symbol
  • Auto-refresh every 30s
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.bot.live_trader import (
    STRATEGY_MAP,
    create_live_session,
    get_active_live_session,
    stop_live_session,
)
from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.database.models import Portfolio as PortfolioModel, SimRun, Trade
from app.database.session import get_session, init_database
from app.strategies.base import Signal

st.set_page_config(
    page_title="Live — TradeFlow",
    layout="wide",
    page_icon="🔴",
)
st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}</style>", unsafe_allow_html=True)

# Auto-refresh every 30 seconds
st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

init_database()

STRATEGY_LABELS = {
    "sma_crossover": "SMA Crossover (20/50)",
    "rsi": "RSI (14) [30/70]",
    "macd": "MACD (12/26/9)",
}
SIGNAL_STYLE = {
    Signal.BUY:  {"icon": "🟢", "label": "ACHETER", "color": "#00C896"},
    Signal.SELL: {"icon": "🔴", "label": "VENDRE",  "color": "#FF4B6E"},
    Signal.HOLD: {"icon": "⏸️", "label": "ATTENDRE", "color": "#8B949E"},
}

DEFAULT_SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "MC.PA", "TTE.PA", "AIR.PA"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔴 Trading en Temps Réel")
st.caption("Le bot surveille les marchés et exécute des trades simulés automatiquement.")

# ── Load active session ────────────────────────────────────────────────────────
active = get_active_live_session()

# ── START / STOP panel ─────────────────────────────────────────────────────────
with st.expander("⚙️ Configurer et démarrer le bot", expanded=(active is None)):
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_key = st.selectbox(
            "Stratégie",
            list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
            key="live_cfg_strategy",
        )
    with col2:
        symbols = st.multiselect(
            "Symboles à trader",
            DEFAULT_SYMBOLS,
            default=["AAPL", "TSLA", "MC.PA"],
            key="live_cfg_symbols",
        )
        custom = st.text_input("Ajouter un ticker personnalisé", placeholder="ex: NVDA", key="live_cfg_custom")
        if custom:
            for t in [x.strip().upper() for x in custom.split(",") if x.strip()]:
                if t not in symbols:
                    symbols.append(t)
    with col3:
        interval = st.selectbox("Intervalle", ["1h", "15m", "1d"], key="live_cfg_interval")
        capital = st.number_input("Capital initial ($)", min_value=1000, max_value=10_000_000,
                                   value=10_000, step=1000, key="live_cfg_capital")

    c_start, c_stop = st.columns(2)
    with c_start:
        if st.button("🚀 Démarrer le bot", use_container_width=True, type="primary",
                     disabled=(active is not None)):
            if not symbols:
                st.error("Choisissez au moins un symbole.")
            else:
                run_id = create_live_session(strategy_key, symbols, interval, capital)
                st.success(f"✅ Session live #{run_id} démarrée ! Le bot commence à la prochaine tick.")
                st.rerun()
    with c_stop:
        if st.button("🛑 Arrêter le bot", use_container_width=True,
                     disabled=(active is None)):
            stop_live_session()
            st.warning("⏹️ Bot arrêté.")
            st.rerun()

st.markdown("---")

# ── No active session ──────────────────────────────────────────────────────────
if active is None:
    st.info("Aucune session active. Configurez et démarrez le bot ci-dessus.")

    # Show past live sessions
    session = get_session()
    try:
        past = session.query(SimRun).filter_by(is_live=True).order_by(SimRun.created_at.desc()).limit(5).all()
    finally:
        session.close()

    if past:
        st.markdown("### Sessions passées")
        past_df = pd.DataFrame([r.to_dict() for r in past])
        cols = ["id", "strategy", "symbol", "interval", "initial_capital",
                "final_value", "total_return_pct", "status", "created_at"]
        st.dataframe(past_df[[c for c in cols if c in past_df.columns]],
                     use_container_width=True, hide_index=True)
    st.stop()

# ── Active session header ──────────────────────────────────────────────────────
run_id = active["id"]
symbols_live = [s.strip() for s in active["symbol"].split(",")]
strategy_live = active["strategy"]
initial_cap = active["initial_capital"]
last_tick = active.get("last_tick_at")

st.markdown(
    f"""
    <div style="background:#0d2b1e;border:1px solid #00C896;border-radius:8px;padding:1rem 1.5rem;margin-bottom:1rem;">
        <span style="font-size:1.1rem;font-weight:700;color:#00C896;">🤖 Bot ACTIF</span>
        &nbsp;·&nbsp;
        <span style="color:#E6EDF3;">Session #{run_id}</span>
        &nbsp;·&nbsp;
        <span style="color:#8B949E;">{STRATEGY_LABELS.get(strategy_live, strategy_live)}</span>
        &nbsp;·&nbsp;
        <span style="color:#8B949E;">{', '.join(symbols_live)}</span>
        &nbsp;·&nbsp;
        <span style="color:#8B949E;">Interval : {active['interval']}</span>
        {"&nbsp;·&nbsp;<span style='color:#8B949E;'>Dernier tick : " + last_tick[:19].replace("T"," ") + "</span>" if last_tick else ""}
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Load live data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_recent_trades(run_id: int) -> pd.DataFrame:
    session = get_session()
    try:
        trades = (
            session.query(Trade)
            .filter_by(sim_run_id=run_id)
            .order_by(Trade.timestamp.desc())
            .limit(50)
            .all()
        )
        return pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_equity_snapshots(run_id: int) -> pd.DataFrame:
    session = get_session()
    try:
        snaps = (
            session.query(PortfolioModel)
            .filter_by(sim_run_id=run_id)
            .order_by(PortfolioModel.timestamp.asc())
            .all()
        )
        if not snaps:
            return pd.DataFrame()
        return pd.DataFrame([{"timestamp": s.timestamp, "total_value": s.total_value, "cash": s.cash} for s in snaps])
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_latest_positions(run_id: int) -> dict:
    session = get_session()
    try:
        snap = (
            session.query(PortfolioModel)
            .filter_by(sim_run_id=run_id)
            .order_by(PortfolioModel.timestamp.desc())
            .first()
        )
        if snap:
            return {"cash": snap.cash, "total_value": snap.total_value,
                    "positions": json.loads(snap.positions_json)}
        return {}
    finally:
        session.close()


trades_df = load_recent_trades(run_id)
equity_df = load_equity_snapshots(run_id)
latest = load_latest_positions(run_id)

# ── Portfolio metrics ──────────────────────────────────────────────────────────
cash = latest.get("cash", initial_cap)
total_value = latest.get("total_value", initial_cap)
positions = latest.get("positions", {})
pnl = total_value - initial_cap
pnl_pct = pnl / initial_cap * 100
n_trades = len(trades_df) if not trades_df.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💼 Valeur totale", f"${total_value:,.2f}", delta=f"{pnl_pct:+.2f}%")
m2.metric("💵 Cash disponible", f"${cash:,.2f}")
m3.metric("📈 P&L total", f"${pnl:+,.2f}")
m4.metric("🔄 Positions ouvertes", len(positions))
m5.metric("📋 Trades exécutés", n_trades)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout: equity curve + positions ────────────────────────────────
col_chart, col_pos = st.columns([2, 1])

with col_chart:
    st.markdown("#### 📈 Évolution du portefeuille")
    if not equity_df.empty:
        color = "#00C896" if total_value >= initial_cap else "#FF4B6E"
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["total_value"],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1,3,5))},0.1)",
            name="Valeur portfolio",
        ))
        fig_eq.add_hline(y=initial_cap, line_dash="dash",
                         line_color="rgba(255,255,255,0.3)",
                         annotation_text=f"Capital initial ${initial_cap:,.0f}")
        fig_eq.update_layout(
            template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            height=280, margin=dict(l=0, r=10, t=20, b=0),
            font=dict(size=12), showlegend=False,
        )
        fig_eq.update_xaxes(gridcolor="#1E2530")
        fig_eq.update_yaxes(gridcolor="#1E2530")
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Aucune donnée encore — le bot n'a pas encore effectué son premier tick.")

with col_pos:
    st.markdown("#### 🗂️ Positions ouvertes")
    if positions:
        for sym, pos in positions.items():
            qty = pos.get("qty", 0)
            avg = pos.get("avg_price", 0)
            cur = pos.get("current_price", avg)
            unrealized = (cur - avg) * qty
            color = "#00C896" if unrealized >= 0 else "#FF4B6E"
            st.markdown(
                f"""
                <div style="border:1px solid #30363D;border-left:4px solid {color};
                            border-radius:6px;padding:0.7rem 1rem;margin-bottom:0.5rem;">
                    <b style="color:#E6EDF3;">{sym}</b>
                    <span style="float:right;color:{color};font-weight:700;">
                        {'+' if unrealized >= 0 else ''}${unrealized:.2f}
                    </span><br>
                    <span style="color:#8B949E;font-size:0.82rem;">
                        {qty:.4f} actions · Coût moy. ${avg:.2f} · Prix actuel ${cur:.2f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Aucune position ouverte pour l'instant.")

st.markdown("---")

# ── Current signals (what the bot sees RIGHT NOW) ──────────────────────────────
st.markdown("#### 🧠 Signaux actuels du bot")
st.caption("Ce que le bot analyserait s'il tournait maintenant sur les données les plus récentes.")

strategy_cls = STRATEGY_MAP.get(strategy_live)
sig_cols = st.columns(len(symbols_live))

for i, symbol in enumerate(symbols_live):
    with sig_cols[i]:
        try:
            df_live = fetch_ohlcv(symbol, interval=active["interval"], period="3mo")
            if df_live is not None and not df_live.empty:
                df_live = add_all_indicators(df_live)
                strategy_inst = strategy_cls() if strategy_cls else None
                if strategy_inst:
                    signal, reason = strategy_inst.generate_signal(df_live, len(df_live) - 1)
                else:
                    signal, reason = Signal.HOLD, "Stratégie inconnue"
                style = SIGNAL_STYLE[signal]
                price = float(df_live.iloc[-1]["close"])
                st.markdown(
                    f"""
                    <div style="border:1px solid {style['color']}33;border-left:4px solid {style['color']};
                                border-radius:8px;padding:0.8rem;text-align:center;">
                        <div style="font-size:1.8rem;">{style['icon']}</div>
                        <div style="font-weight:700;color:{style['color']};font-size:1.1rem;">{style['label']}</div>
                        <div style="color:#E6EDF3;font-size:0.9rem;margin-top:2px;">{symbol} — ${price:.2f}</div>
                        <div style="color:#8B949E;font-size:0.78rem;margin-top:6px;font-style:italic;">{reason or 'En attente d\'un signal'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"Données indisponibles pour {symbol}")
        except Exception as e:
            st.error(f"{symbol}: {e}")

st.markdown("---")

# ── Recent trades ──────────────────────────────────────────────────────────────
st.markdown("#### 📋 Derniers trades exécutés")

if trades_df.empty:
    st.info("Aucun trade exécuté pour l'instant. Le bot attend le prochain signal.")
else:
    display = trades_df.copy()
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    if "pnl" in display.columns:
        display["pnl"] = display["pnl"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
    if "price" in display.columns:
        display["price"] = display["price"].apply(lambda x: f"${x:.2f}")
    if "fees" in display.columns:
        display["fees"] = display["fees"].apply(lambda x: f"${x:.4f}")
    if "quantity" in display.columns:
        display["quantity"] = display["quantity"].apply(lambda x: f"{x:.4f}")

    col_map = {
        "timestamp": "Date/Heure", "symbol": "Symbole", "side": "Sens",
        "quantity": "Quantité", "price": "Prix", "fees": "Frais",
        "pnl": "P&L", "reason": "🤖 Raison",
    }
    cols_show = [c for c in ["timestamp", "symbol", "side", "quantity", "price", "fees", "pnl", "reason"]
                 if c in display.columns]
    display = display[cols_show].rename(columns=col_map)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "🤖 Raison": st.column_config.TextColumn("🤖 Raison", width="large"),
            "Sens": st.column_config.TextColumn("Sens", width="small"),
        },
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:right;color:#8B949E;font-size:0.75rem;margin-top:1rem;'>"
    f"Rafraîchissement auto toutes les 30s · "
    f"Dernière vue : {datetime.now().strftime('%H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
