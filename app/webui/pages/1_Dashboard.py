"""
TradeFlow — Dashboard Page
Portfolio overview: equity curve, key metrics, and simulation history table.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st

from app.database.models import Portfolio as PortfolioModel, SimRun, Trade
from app.database.session import get_session, init_database
from app.webui.components.charts import build_equity_curve_chart, build_returns_distribution
from app.webui.components.metrics import (
    compute_average_trade_pnl,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_win_rate,
)

st.set_page_config(page_title="Tableau de Bord — TradeFlow", layout="wide", page_icon="📊")

# ── Inject shared CSS ─────────────────────────────────────────────────────────
st.markdown(
    "<style>#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()


@st.cache_data(ttl=30)
def load_all_sim_runs() -> pd.DataFrame:
    """Load all simulation runs from the database."""
    session = get_session()
    try:
        runs = session.query(SimRun).order_by(SimRun.created_at.desc()).all()
        return pd.DataFrame([r.to_dict() for r in runs]) if runs else pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=30)
def load_equity_curve(sim_run_id: int) -> pd.DataFrame:
    """Load portfolio snapshots for a given simulation run."""
    session = get_session()
    try:
        snaps = (
            session.query(PortfolioModel)
            .filter(PortfolioModel.sim_run_id == sim_run_id)
            .order_by(PortfolioModel.timestamp.asc())
            .all()
        )
        if not snaps:
            return pd.DataFrame()
        return pd.DataFrame([{"timestamp": s.timestamp, "total_value": s.total_value} for s in snaps])
    finally:
        session.close()


@st.cache_data(ttl=30)
def load_trades(sim_run_id: int) -> pd.DataFrame:
    """Load all trades for a given simulation run."""
    session = get_session()
    try:
        trades = (
            session.query(Trade)
            .filter(Trade.sim_run_id == sim_run_id)
            .order_by(Trade.timestamp.asc())
            .all()
        )
        return pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    finally:
        session.close()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Tableau de Bord")
st.markdown("---")

runs_df = load_all_sim_runs()

if runs_df.empty:
    st.info("Aucune simulation trouvée. Allez dans **Simulation** pour lancer votre premier backtest.")
    st.stop()

# ── Simulation selector ───────────────────────────────────────────────────────
col_select, col_refresh = st.columns([5, 1])
with col_select:
    run_options = {
        f"#{row['id']} — {row['strategy']} on {row['symbol']} [{row['interval']}] ({row['created_at'][:10]})": row["id"]
        for _, row in runs_df.iterrows()
    }
    selected_label = st.selectbox("Sélectionner une simulation", options=list(run_options.keys()), key="dash_run_select")
    selected_run_id = run_options[selected_label]

with col_refresh:
    if st.button("🔄 Rafraîchir", key="dash_refresh"):
        st.cache_data.clear()
        st.rerun()

selected_run = runs_df[runs_df["id"] == selected_run_id].iloc[0]

# ── Key metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)

final_val = selected_run.get("final_value") or 0
initial_cap = selected_run.get("initial_capital") or 10000
ret_pct = selected_run.get("total_return_pct") or 0
sharpe = selected_run.get("sharpe_ratio") or 0
drawdown = selected_run.get("max_drawdown_pct") or 0
win_rate = selected_run.get("win_rate") or 0

with m1:
    st.metric("Valeur du Portefeuille", f"${final_val:,.2f}", delta=f"{ret_pct:+.2f}%")
with m2:
    st.metric("Rendement Total", f"{ret_pct:+.2f}%")
with m3:
    st.metric("Ratio de Sharpe", f"{sharpe:.2f}")
with m4:
    st.metric("Drawdown Max", f"-{drawdown:.2f}%")
with m5:
    st.metric("Taux de Réussite", f"{win_rate * 100:.1f}%")
with m6:
    st.metric("Total des Trades", str(selected_run.get("total_trades") or 0))

st.markdown("<br>", unsafe_allow_html=True)

# ── Equity curve ──────────────────────────────────────────────────────────────
equity_df = load_equity_curve(selected_run_id)

col_chart, col_info = st.columns([3, 1])
with col_chart:
    if not equity_df.empty:
        fig = build_equity_curve_chart(equity_df, initial_capital=initial_cap)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée de courbe de capital pour cette simulation.")

with col_info:
    st.markdown(
        f"""
        <div style="padding:1rem; border-radius:10px; background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-top:0;">Détails de la simulation</h4>
            <b>Stratégie :</b> <span style="color:#00C896;">{selected_run.get('strategy','—')}</span><br>
            <b>Symbole :</b> {selected_run.get('symbol','—')}<br>
            <b>Intervalle :</b> {selected_run.get('interval','—')}<br>
            <b>Période :</b> {selected_run.get('start_date','—')} → {selected_run.get('end_date','—')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # P&L distribution
    trades_df = load_trades(selected_run_id)
    if not trades_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        fig_dist = build_returns_distribution(trades_df)
        st.plotly_chart(fig_dist, use_container_width=True)

# ── All simulations table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Toutes les Simulations")

display_cols = ["id", "strategy", "symbol", "interval", "initial_capital",
                "final_value", "total_return_pct", "sharpe_ratio",
                "max_drawdown_pct", "win_rate", "total_trades", "created_at"]
available_cols = [c for c in display_cols if c in runs_df.columns]
display_df = runs_df[available_cols].copy()

# Format percentage columns
for pct_col in ["total_return_pct", "max_drawdown_pct", "win_rate"]:
    if pct_col in display_df.columns:
        display_df[pct_col] = display_df[pct_col].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
        )

st.dataframe(display_df, use_container_width=True, hide_index=True)
