"""
TradeFlow — Analyse en Direct
Montre l'état actuel du bot pour chaque symbole :
indicateurs en temps réel + signal que le bot émettrait MAINTENANT.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from app.data.fetcher import fetch_ohlcv
from app.data.indicators import add_all_indicators
from app.strategies.sma_crossover import SmaCrossoverStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.base import Signal

st.set_page_config(page_title="Analyse en Direct — TradeFlow", layout="wide", page_icon="🔴")
st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}</style>", unsafe_allow_html=True)

SYMBOLS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "LVMH (MC.PA)": "MC.PA",
    "TotalEnergies (TTE.PA)": "TTE.PA",
    "Airbus (AIR.PA)": "AIR.PA",
}

STRATEGIES = {
    "SMA Crossover (20/50)": SmaCrossoverStrategy,
    "RSI (14) [30/70]": RsiStrategy,
    "MACD (12/26/9)": MacdStrategy,
}

SIGNAL_COLORS = {
    Signal.BUY:  {"bg": "#0d2b1e", "border": "#00C896", "text": "#00C896", "icon": "🟢", "label": "ACHETER"},
    Signal.SELL: {"bg": "#2b0d12", "border": "#FF4B6E", "text": "#FF4B6E", "icon": "🔴", "label": "VENDRE"},
    Signal.HOLD: {"bg": "#1a1a2e", "border": "#58A6FF", "text": "#8B949E", "icon": "⏸️", "label": "ATTENDRE"},
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔴 Analyse en Direct")
st.markdown("Signaux que le bot émettrait **maintenant** sur les dernières données disponibles.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paramètres")
    interval = st.selectbox("Intervalle", ["1h", "15m", "1d"], key="live_interval")
    period = st.selectbox("Fenêtre d'analyse", ["1mo", "3mo", "6mo"], index=1, key="live_period",
                          format_func=lambda x: {"1mo": "1 mois", "3mo": "3 mois", "6mo": "6 mois"}[x])
    selected_symbols = st.multiselect(
        "Symboles à analyser",
        list(SYMBOLS.keys()),
        default=["Apple (AAPL)", "Tesla (TSLA)", "LVMH (MC.PA)"],
        key="live_symbols",
    )
    selected_strats = st.multiselect(
        "Stratégies",
        list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        key="live_strats",
    )
    refresh = st.button("🔄 Actualiser les données", use_container_width=True)

if not selected_symbols:
    st.warning("Sélectionnez au moins un symbole dans le menu latéral.")
    st.stop()

if not selected_strats:
    st.warning("Sélectionnez au moins une stratégie.")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def load_live_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = fetch_ohlcv(symbol, interval=interval, period=period, use_cache=False)
    if df is not None and not df.empty:
        return add_all_indicators(df)
    return pd.DataFrame()


def get_current_indicators(df: pd.DataFrame) -> dict:
    """Extract the most recent indicator values."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    result = {"close": float(last["close"])}

    rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
    if rsi_cols:
        result["rsi"] = float(last[rsi_cols[0]]) if not pd.isna(last[rsi_cols[0]]) else None

    sma_20 = "sma_20"
    sma_50 = "sma_50"
    if sma_20 in df.columns and not pd.isna(last[sma_20]):
        result["sma_20"] = float(last[sma_20])
    if sma_50 in df.columns and not pd.isna(last[sma_50]):
        result["sma_50"] = float(last[sma_50])

    macd_cols = [c for c in df.columns if c.startswith("MACD_") and "MACDs" not in c and "MACDh" not in c]
    signal_cols = [c for c in df.columns if c.startswith("MACDs_")]
    if macd_cols and not pd.isna(last[macd_cols[0]]):
        result["macd"] = float(last[macd_cols[0]])
    if signal_cols and not pd.isna(last[signal_cols[0]]):
        result["macd_signal"] = float(last[signal_cols[0]])

    return result


def render_signal_card(symbol: str, strategy_name: str, signal: Signal, reason: str, indicators: dict):
    """Render a signal card with color coding and indicator values."""
    style = SIGNAL_COLORS[signal]
    price = indicators.get("close", 0)

    # Build indicator badges
    badges = []
    if "rsi" in indicators and indicators["rsi"] is not None:
        rsi = indicators["rsi"]
        rsi_color = "#FF4B6E" if rsi > 70 else ("#00C896" if rsi < 30 else "#8B949E")
        badges.append(f'<span style="background:{rsi_color}22;color:{rsi_color};padding:2px 8px;border-radius:4px;font-size:0.75rem;">RSI {rsi:.1f}</span>')
    if "sma_20" in indicators and "sma_50" in indicators:
        diff = indicators["sma_20"] - indicators["sma_50"]
        sma_color = "#00C896" if diff > 0 else "#FF4B6E"
        badges.append(f'<span style="background:{sma_color}22;color:{sma_color};padding:2px 8px;border-radius:4px;font-size:0.75rem;">SMA20{"↑" if diff > 0 else "↓"}SMA50</span>')
    if "macd" in indicators and "macd_signal" in indicators:
        diff = indicators["macd"] - indicators["macd_signal"]
        macd_color = "#00C896" if diff > 0 else "#FF4B6E"
        badges.append(f'<span style="background:{macd_color}22;color:{macd_color};padding:2px 8px;border-radius:4px;font-size:0.75rem;">MACD{"↑" if diff > 0 else "↓"}</span>')

    badges_html = " ".join(badges)
    reason_html = f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#8B949E;font-style:italic;">{reason if reason else "Pas de signal actif — en attente"}</div>' if signal == Signal.HOLD else f'<div style="margin-top:0.5rem;font-size:0.85rem;color:{style["text"]};">{reason}</div>'

    st.markdown(
        f"""
        <div style="
            background:{style['bg']};
            border:1px solid {style['border']};
            border-left:4px solid {style['border']};
            border-radius:8px;
            padding:1rem 1.2rem;
            margin-bottom:0.75rem;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:1.1rem;font-weight:700;color:#E6EDF3;">{style['icon']} {symbol}</span>
                    <span style="margin-left:0.5rem;font-size:0.8rem;color:#8B949E;">{strategy_name}</span>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:1.3rem;font-weight:800;color:{style['text']};">{style['label']}</span>
                    <div style="font-size:0.85rem;color:#8B949E;">${price:.2f}</div>
                </div>
            </div>
            <div style="margin-top:0.6rem;">{badges_html}</div>
            {reason_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main analysis loop ────────────────────────────────────────────────────────
if refresh:
    st.cache_data.clear()

st.markdown(f"*Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')} — données {interval}*")
st.markdown("<br>", unsafe_allow_html=True)

# Summary counters
total_buy = 0
total_sell = 0
total_hold = 0

all_results = []

for sym_label in selected_symbols:
    symbol = SYMBOLS[sym_label]

    with st.spinner(f"Chargement {symbol}…"):
        df = load_live_data(symbol, interval, period)

    if df.empty:
        st.error(f"❌ Impossible de charger les données pour **{symbol}**")
        continue

    indicators = get_current_indicators(df)
    last_idx = len(df) - 1

    for strat_name in selected_strats:
        strategy = STRATEGIES[strat_name]()
        signal, reason = strategy.generate_signal(df, last_idx)

        if signal == Signal.BUY:
            total_buy += 1
        elif signal == Signal.SELL:
            total_sell += 1
        else:
            total_hold += 1

        all_results.append({
            "symbol": symbol,
            "strategy": strat_name,
            "signal": signal,
            "reason": reason,
            "indicators": indicators,
        })

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🟢 Signaux ACHAT", total_buy)
with c2:
    st.metric("🔴 Signaux VENTE", total_sell)
with c3:
    st.metric("⏸️ En Attente", total_hold)
with c4:
    total = total_buy + total_sell + total_hold
    pct_active = round((total_buy + total_sell) / total * 100, 1) if total > 0 else 0
    st.metric("Signaux actifs", f"{pct_active}%")

st.markdown("---")

# ── Signal cards by symbol ────────────────────────────────────────────────────
# Group by symbol, show 3 columns
sym_groups = {}
for r in all_results:
    sym_groups.setdefault(r["symbol"], []).append(r)

n_cols = min(3, len(sym_groups))
if n_cols == 0:
    st.stop()

cols = st.columns(n_cols)
for i, (symbol, results) in enumerate(sym_groups.items()):
    with cols[i % n_cols]:
        st.markdown(f"#### {symbol}")
        close_price = results[0]["indicators"].get("close", 0) if results else 0

        # Mini price chart
        df_sym = load_live_data(symbol, interval, period)
        if not df_sym.empty and len(df_sym) > 10:
            recent = df_sym.tail(48)  # Last 48 bars
            color = "#00C896" if recent["close"].iloc[-1] >= recent["close"].iloc[0] else "#FF4B6E"
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(
                x=recent.index,
                y=recent["close"],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"{'rgba(0,200,150,0.1)' if color == '#00C896' else 'rgba(255,75,110,0.1)'}",
            ))
            fig_mini.update_layout(
                height=120,
                margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_mini, use_container_width=True)

        for r in results:
            render_signal_card(
                symbol=r["symbol"],
                strategy_name=r["strategy"],
                signal=r["signal"],
                reason=r["reason"],
                indicators=r["indicators"],
            )

# ── Tableau récap ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Récapitulatif")

if all_results:
    recap_df = pd.DataFrame([{
        "Symbole": r["symbol"],
        "Stratégie": r["strategy"],
        "Signal": f"{SIGNAL_COLORS[r['signal']]['icon']} {SIGNAL_COLORS[r['signal']]['label']}",
        "Prix actuel": f"${r['indicators'].get('close', 0):.2f}",
        "RSI": f"{r['indicators']['rsi']:.1f}" if r['indicators'].get('rsi') else "—",
        "Raison": r["reason"] or "En attente",
    } for r in all_results])

    st.dataframe(recap_df, use_container_width=True, hide_index=True,
                 column_config={"Raison": st.column_config.TextColumn("Raison", width="large")})
