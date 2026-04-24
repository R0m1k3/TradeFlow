"""
TradeFlow — Trades Page
Full trade history with performance statistics and P&L analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st

from app.database.models import SimRun, Trade
from app.database.session import get_session, init_database
from app.webui.components.charts import build_returns_distribution
from app.webui.components.metrics import (
    compute_average_trade_pnl,
    compute_profit_factor,
    compute_win_rate,
)

st.set_page_config(page_title="Transactions — TradeFlow", layout="wide", page_icon="📋")
st.markdown(
    "<style>#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()

st.markdown("## 📋 Historique des Transactions")
st.markdown("---")

# ── Load simulation runs ───────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    session = get_session()
    try:
        runs = session.query(SimRun).order_by(SimRun.created_at.desc()).all()
        return pd.DataFrame([r.to_dict() for r in runs]) if runs else pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=30)
def load_trades_for_run(run_id: int) -> pd.DataFrame:
    session = get_session()
    try:
        trades = (
            session.query(Trade)
            .filter(Trade.sim_run_id == run_id)
            .order_by(Trade.timestamp.asc())
            .all()
        )
        return pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    finally:
        session.close()


runs_df = load_runs()

if runs_df.empty:
    st.info("Aucune simulation trouvée. Allez dans **Simulation** pour lancer votre premier backtest.")
    st.stop()

# ── Selectors ─────────────────────────────────────────────────────────────────
col_sel, col_side = st.columns([4, 1])
with col_sel:
    run_options = {
        f"#{r['id']} — {r['strategy']} on {r['symbol']} [{r['interval']}] ({r['created_at'][:10]})": r["id"]
        for _, r in runs_df.iterrows()
    }
    selected_label = st.selectbox("Sélectionner une simulation", list(run_options.keys()), key="trades_run_sel")
    run_id = run_options[selected_label]

with col_side:
    side_filter = st.selectbox("Filtrer par sens (Side)", ["ALL", "BUY", "SELL"], key="trades_side_filter")

trades_df = load_trades_for_run(run_id)

if trades_df.empty:
    st.warning("Aucune transaction enregistrée pour cette simulation.")
    st.stop()

# Apply side filter
if side_filter != "ALL":
    filtered_df = trades_df[trades_df["side"] == side_filter]
else:
    filtered_df = trades_df

# ── Performance metrics ───────────────────────────────────────────────────────
win_rate, winning, total_sells = compute_win_rate(trades_df)
profit_factor = compute_profit_factor(trades_df)
avg_pnl = compute_average_trade_pnl(trades_df)
total_pnl = trades_df[trades_df["side"] == "SELL"]["pnl"].sum() if not trades_df.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total des Trades", len(trades_df))
with m2:
    st.metric("Taux de Réussite", f"{win_rate * 100:.1f}%")
with m3:
    st.metric("Facteur de Profit", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")
with m4:
    st.metric("P&L Moyen par Trade", f"${avg_pnl:+.2f}")
with m5:
    st.metric("P&L Réalisé Total", f"${total_pnl:+.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
col_dist, col_cumulative = st.columns(2)

with col_dist:
    fig_dist = build_returns_distribution(trades_df)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_cumulative:
    # Cumulative P&L chart
    import plotly.graph_objects as go

    sell_trades = trades_df[trades_df["side"] == "SELL"].copy()
    if not sell_trades.empty:
        sell_trades["timestamp"] = pd.to_datetime(sell_trades["timestamp"])
        sell_trades = sell_trades.sort_values("timestamp")
        sell_trades["cum_pnl"] = sell_trades["pnl"].cumsum()

        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=sell_trades["timestamp"],
                y=sell_trades["cum_pnl"],
                mode="lines+markers",
                name="P&L Cumulé",
                line=dict(color="#00C896", width=2),
                marker=dict(
                    color=["#00C896" if p >= 0 else "#FF4B6E" for p in sell_trades["pnl"]],
                    size=6,
                ),
                fill="tozeroy",
                fillcolor="rgba(0,200,150,0.08)",
            )
        )
        fig_cum.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_cum.update_layout(
            title="P&L Cumulé",
            template="plotly_dark",
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            height=300,
            margin=dict(l=0, r=10, t=50, b=0),
            font=dict(family="Inter, sans-serif", size=12),
        )
        fig_cum.update_xaxes(gridcolor="#1E2530")
        fig_cum.update_yaxes(gridcolor="#1E2530")
        st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---")

# ── Trades table ──────────────────────────────────────────────────────────────
st.markdown(f"### Transactions ({len(filtered_df)} résultats)")

# Color-code P&L in display
display_df = filtered_df.copy()
if "timestamp" in display_df.columns:
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
if "pnl" in display_df.columns:
    display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
if "price" in display_df.columns:
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:.4f}")
if "fees" in display_df.columns:
    display_df["fees"] = display_df["fees"].apply(lambda x: f"${x:.4f}")
if "quantity" in display_df.columns:
    display_df["quantity"] = display_df["quantity"].apply(lambda x: f"{x:.4f}")

cols_order = ["id", "timestamp", "symbol", "side", "quantity", "price", "fees", "pnl", "reason"]
cols_display = [c for c in cols_order if c in display_df.columns]

col_labels = {
    "id": "ID", "timestamp": "Date/Heure", "symbol": "Symbole", "side": "Sens",
    "quantity": "Quantité", "price": "Prix", "fees": "Frais", "pnl": "P&L",
    "reason": "🤖 Pourquoi le bot a agi",
}
display_df = display_df[cols_display].rename(columns=col_labels)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "🤖 Pourquoi le bot a agi": st.column_config.TextColumn(
            "🤖 Pourquoi le bot a agi",
            width="large",
            help="Explication de la stratégie au moment du signal",
        ),
        "Sens": st.column_config.TextColumn("Sens", width="small"),
        "P&L": st.column_config.TextColumn("P&L", width="small"),
    },
)
