"""
TradeFlow — Charts Page
Interactive candlestick charts with technical indicators and trade markers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.database.models import Trade, SimRun
from app.database.session import get_session, init_database
from app.webui.components.charts import build_candlestick_chart

st.set_page_config(page_title="Graphiques — TradeFlow", layout="wide", page_icon="🕯️")
st.markdown(
    "<style>html,body,[class*='css']{font-family:'Inter',sans-serif!important;}"
    ".main .block-container{padding:1.5rem 2rem;max-width:1600px;}"
    "#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()

DEFAULT_SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "MC.PA", "TTE.PA"]
INTERVALS = ["1h", "1d", "15m", "30m"]
PERIODS = {"1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}

st.markdown("## 🕯️ Graphiques")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paramètres du Graphique")
    symbol = st.selectbox("Symbole", DEFAULT_SYMBOLS, key="chart_symbol")
    custom_symbol = st.text_input("Ou entrez un ticker personnalisé", placeholder="ex: NVDA", key="chart_custom")
    if custom_symbol:
        symbol = custom_symbol.upper().strip()

    interval = st.selectbox("Intervalle", INTERVALS, key="chart_interval")
    period_label = st.selectbox("Période", ["1 Semaine", "1 Mois", "3 Mois", "6 Mois", "1 An"], index=2, key="chart_period")
    
    # Map French labels back to yfinance period strings
    fr_periods = {"1 Semaine": "5d", "1 Mois": "1mo", "3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y"}
    period = fr_periods[period_label]

    st.markdown("### Superpositions")
    show_sma = st.toggle("SMA (20 / 50 / 200)", value=True, key="chart_sma")
    show_bb = st.toggle("Bandes de Bollinger", value=True, key="chart_bb")

    st.markdown("### Marqueurs de transactions")
    show_trades = st.toggle("Superposer les marqueurs de transactions", value=True, key="chart_trades")

    # Sim run selector for trade markers
    trades_df = pd.DataFrame()
    if show_trades:
        session = get_session()
        try:
            runs = (
                session.query(SimRun)
                .filter(SimRun.symbol == symbol)
                .order_by(SimRun.created_at.desc())
                .limit(10)
                .all()
            )
        finally:
            session.close()

        if runs:
            run_opts = {
                f"#{r.id} {r.strategy} [{r.interval}]": r.id for r in runs
            }
            sel_run_label = st.selectbox("Exécution de la simulation", list(run_opts.keys()), key="chart_run")
            sel_run_id = run_opts[sel_run_label]

            session = get_session()
            try:
                t_rows = (
                    session.query(Trade)
                    .filter(Trade.sim_run_id == sel_run_id)
                    .all()
                )
                trades_df = pd.DataFrame([t.to_dict() for t in t_rows]) if t_rows else pd.DataFrame()
            finally:
                session.close()
        else:
            st.caption("Aucune simulation trouvée pour ce symbole.")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Récupération des données de prix…")
def load_chart_data(sym: str, intv: str, per: str) -> pd.DataFrame:
    """Load and enrich OHLCV data with all indicators."""
    df = fetch_ohlcv(sym, interval=intv, period=per, use_cache=False)
    if df is not None and not df.empty:
        return add_all_indicators(df)
    return pd.DataFrame()


df = load_chart_data(symbol, interval, period)

if df.empty:
    st.error(f"❌ Impossible de charger les données pour **{symbol}** [{interval}]. Vérifiez le ticker ou essayez une période différente.")
    st.stop()

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = build_candlestick_chart(
    df=df,
    symbol=symbol,
    trades_df=trades_df if show_trades and not trades_df.empty else None,
    show_sma=show_sma,
    show_bollinger=show_bb,
)

st.plotly_chart(fig, use_container_width=True)

# ── Quick stats bar ───────────────────────────────────────────────────────────
last = df.iloc[-1]
first = df.iloc[0]
price_change = last["close"] - first["close"]
pct_change = (price_change / first["close"]) * 100
color = "#00C896" if price_change >= 0 else "#FF4B6E"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Dernière Clôture", f"${last['close']:.2f}")
with c2:
    st.metric("Variation sur la période", f"{pct_change:+.2f}%")
with c3:
    st.metric("Plus Haut", f"${df['high'].max():.2f}")
with c4:
    st.metric("Plus Bas", f"${df['low'].min():.2f}")
with c5:
    st.metric("Bougies Chargées", str(len(df)))
