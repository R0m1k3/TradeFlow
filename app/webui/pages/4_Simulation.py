"""
TradeFlow — Simulation Page
Configure and launch backtests with real-time progress tracking.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import time
from typing import Optional

import pandas as pd
import streamlit as st

from app.database.session import init_database
from app.simulator.engine import SimResult, SimulationEngine
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.sma_crossover import SmaCrossoverStrategy
from app.webui.components.charts import build_equity_curve_chart

st.set_page_config(page_title="Simulation — TradeFlow", layout="wide", page_icon="🚀")
st.markdown(
    "<style>#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()

# ── Constants ─────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict[str, type] = {
    "SMA Crossover (20/50)": SmaCrossoverStrategy,
    "RSI (14) [30/70]": RsiStrategy,
    "MACD (12/26/9)": MacdStrategy,
DEFAULT_SYMBOLS = {
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "LVMH (MC.PA)": "MC.PA",
    "TotalEnergies (TTE.PA)": "TTE.PA",
    "Airbus (AIR.PA)": "AIR.PA",
    "BNP Paribas (BNP.PA)": "BNP.PA",
    "Sanofi (SAN.PA)": "SAN.PA",
}
PERIODS = {
    "1 Mois": "1mo",
    "3 Mois": "3mo",
    "6 Mois": "6mo",
    "1 An": "1y",
    "2 Ans": "2y",
}

st.markdown("## 🚀 Simulation")
st.markdown("---")

# ── Configuration form ────────────────────────────────────────────────────────
with st.form("simulation_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Actif et Stratégie")
        symbol_label = st.selectbox("Société / Action", list(DEFAULT_SYMBOLS.keys()), key="sim_symbol")
        symbol = DEFAULT_SYMBOLS[symbol_label]
        custom_sym = st.text_input("Ou ticker personnalisé (ex: GOOG)", placeholder="Laissez vide pour utiliser le choix", key="sim_custom")
        strategy_name = st.selectbox("Stratégie", list(STRATEGY_REGISTRY.keys()), key="sim_strategy")

    with col2:
        st.markdown("### ⏱️ Période et Capital")
        interval_label = st.selectbox("Intervalle (Bougies)", list(INTERVALS.keys()), key="sim_interval")
        period_label = st.selectbox("Période historique", list(PERIODS.keys()), index=1, key="sim_period")
        initial_capital = st.number_input(
            "Capital initial ($)", min_value=1000, max_value=10_000_000,
            value=10_000, step=1000, key="sim_capital"
        )
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🏦 Paramètres du Courtier")
        commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01, key="sim_commission") / 100
        slippage = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.005, key="sim_slippage") / 100
    with col4:
        st.markdown("### ⚙️ Gestion du risque")
        position_size = st.slider("Taille de la position (%)", 10, 100, 95, 5, key="sim_pos_size") / 100

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-strategy comparison
    st.markdown("### Comparaison Multi-Stratégies *(optionnel)*")
    run_comparison = st.checkbox(
        "Lancer les 3 stratégies sur le même symbole et comparer",
        value=False,
        key="sim_compare",
    )

    submitted = st.form_submit_button("🚀 Lancer la Simulation", use_container_width=True)

# ── Run simulation ─────────────────────────────────────────────────────────────
if submitted:
    effective_symbol = custom_sym.upper().strip() if custom_sym else symbol
    interval = INTERVALS[interval_label]
    period = PERIODS[period_label]

    strategies_to_run = (
        list(STRATEGY_REGISTRY.items()) if run_comparison
        else [(strategy_name, STRATEGY_REGISTRY[strategy_name])]
    )

    results: list[SimResult] = []
    engine = SimulationEngine(
        commission_rate=commission,
        slippage_rate=slippage,
        position_size_pct=position_size,
    )

    for strat_name, strat_cls in strategies_to_run:
        st.markdown(f"---\n#### Exécution de : **{strat_name}** sur `{effective_symbol}`")

        progress_bar = st.progress(0, text=f"Initialisation de {strat_name}…")
        status_placeholder = st.empty()

        def _update_progress(pct: float) -> None:
            progress_bar.progress(min(pct, 1.0), text=f"Traitement des bougies… {pct*100:.0f}%")

        # Instantiate strategy with default params
        strategy = strat_cls()

        with st.spinner(f"Récupération des données et exécution du backtest…"):
            result = engine.run(
                strategy=strategy,
                symbol=effective_symbol,
                interval=interval,
                period=period,
                initial_capital=initial_capital,
                progress_callback=_update_progress,
            )

        progress_bar.progress(1.0, text="✅ Terminé")

        if result is None:
            st.error(
                f"❌ La simulation a échoué pour **{strat_name}** sur **{effective_symbol}**. "
                "Vérifiez le symbole et l'intervalle."
            )
            continue

        results.append(result)

        # ── Results display ────────────────────────────────────────────────
        st.success(f"✅ Simulation terminée pour {strat_name} !")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Valeur Finale", f"${result.final_value:,.2f}")
        m2.metric("Rendement", f"{result.total_return_pct:+.2f}%")
        m3.metric("Ratio Sharpe", f"{result.sharpe_ratio:.2f}")
        m4.metric("Drawdown Max", f"-{result.max_drawdown_pct:.2f}%")
        m5.metric("Taux de réussite", f"{result.win_rate * 100:.1f}%")
        m6.metric("Total Trades", result.total_trades)

        # Equity curve for this run
        if result.equity_curve:
            equity_df = pd.DataFrame(result.equity_curve, columns=["timestamp", "total_value"])
            fig = build_equity_curve_chart(equity_df, initial_capital, title=f"Courbe de Capital — {strat_name}")
            st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    if run_comparison and len(results) > 1:
        st.markdown("---")
        st.markdown("### 🏆 Comparaison des Stratégies")
        comp_data = [
            {
                "Stratégie": r.strategy_name,
                "Rendement (%)": f"{r.total_return_pct:+.2f}%",
                "Valeur Finale": f"${r.final_value:,.2f}",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "Drawdown Max": f"-{r.max_drawdown_pct:.2f}%",
                "Taux de réussite": f"{r.win_rate * 100:.1f}%",
                "Trades": r.total_trades,
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    if results:
        st.success(f"✅ {len(results)} simulation(s) terminée(s) et sauvegardée(s) en base de données.")
        st.info("📊 Voir les résultats sur les pages **Tableau de Bord** et **Transactions**.")
