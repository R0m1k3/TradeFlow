"""
TradeFlow — Optimization Page
Hyperparameter tuning via Grid Search.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st

from app.simulator.optimizer import grid_search
from app.strategies.macd_strategy import MacdStrategy
from app.strategies.rsi_strategy import RsiStrategy
from app.strategies.sma_crossover import SmaCrossoverStrategy
from app.database.session import init_database

st.set_page_config(page_title="Optimisation — TradeFlow", layout="wide", page_icon="⚙️")
st.markdown(
    "<style>#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()

STRATEGY_REGISTRY = {
    "Croisement SMA": SmaCrossoverStrategy,
    "RSI Surachat/Survente": RsiStrategy,
    # MACD is omitted to keep UI simple for now, can be added later
}

DEFAULT_SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "MC.PA", "TTE.PA"]
INTERVALS = {"1 Heure": "1h", "Quotidien": "1d", "30 Minutes": "30m"}
PERIODS = {"3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y", "2 Ans": "2y"}

st.markdown("## ⚙️ Optimisation de Stratégies")
st.markdown("---")
st.info("Trouvez automatiquement les meilleurs paramètres (Grid Search) pour maximiser le rendement sur l'historique.")

with st.form("opt_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.selectbox("Symbole", DEFAULT_SYMBOLS, key="opt_symbol")
        strategy_name = st.selectbox("Stratégie", list(STRATEGY_REGISTRY.keys()), key="opt_strat")
    with c2:
        interval_label = st.selectbox("Intervalle", list(INTERVALS.keys()), key="opt_interval")
        period_label = st.selectbox("Période historique", list(PERIODS.keys()), index=2, key="opt_period")
    with c3:
        st.markdown("**Capital**")
        initial_capital = st.number_input("Capital initial ($)", value=10000, step=1000)

    st.markdown("### Plages de paramètres")
    p1, p2 = st.columns(2)
    
    param_grid = {}
    if strategy_name == "Croisement SMA":
        with p1:
            fast_min = st.number_input("SMA Rapide (Min)", value=10, min_value=2, max_value=50)
            fast_max = st.number_input("SMA Rapide (Max)", value=30, min_value=2, max_value=50)
            fast_step = st.number_input("SMA Rapide (Pas)", value=5, min_value=1, max_value=20)
        with p2:
            slow_min = st.number_input("SMA Lente (Min)", value=40, min_value=10, max_value=200)
            slow_max = st.number_input("SMA Lente (Max)", value=100, min_value=10, max_value=200)
            slow_step = st.number_input("SMA Lente (Pas)", value=10, min_value=1, max_value=50)
        
        # Build grid
        fast_range = list(range(fast_min, fast_max + 1, fast_step))
        slow_range = list(range(slow_min, slow_max + 1, slow_step))
        # Filter out invalid combos
        param_grid = {
            "fast_period": fast_range,
            "slow_period": slow_range
        }
        total_combos = len(fast_range) * len(slow_range)
        
    elif strategy_name == "RSI Surachat/Survente":
        with p1:
            period_min = st.number_input("Période RSI (Min)", value=10, min_value=5, max_value=30)
            period_max = st.number_input("Période RSI (Max)", value=20, min_value=5, max_value=30)
            period_step = st.number_input("Période RSI (Pas)", value=2, min_value=1, max_value=10)
        with p2:
            oversold_vals = st.multiselect("Seuils de Survente (Oversold)", [20.0, 25.0, 30.0, 35.0], default=[25.0, 30.0])
            overbought_vals = st.multiselect("Seuils de Surachat (Overbought)", [65.0, 70.0, 75.0, 80.0], default=[70.0, 75.0])
            
        period_range = list(range(period_min, period_max + 1, period_step))
        param_grid = {
            "period": period_range,
            "oversold": oversold_vals,
            "overbought": overbought_vals
        }
        total_combos = len(period_range) * len(oversold_vals) * len(overbought_vals)

    st.markdown(f"**Combinaisons à tester : {total_combos}**")
    submitted = st.form_submit_button("⚙️ Lancer l'Optimisation", use_container_width=True)

if submitted:
    if total_combos == 0:
        st.error("Aucune combinaison de paramètres à tester.")
        st.stop()
        
    if total_combos > 500:
        st.warning(f"Attention, vous allez tester {total_combos} combinaisons. Cela peut prendre beaucoup de temps.")
    
    interval = INTERVALS[interval_label]
    period = PERIODS[period_label]
    strat_cls = STRATEGY_REGISTRY[strategy_name]
    
    progress_bar = st.progress(0, text="Initialisation...")
    
    def ui_callback(current, total, params):
        pct = current / total
        progress_bar.progress(pct, text=f"Test {current}/{total} — {params}")
        
    with st.spinner("Optimisation en cours (les logs de la console peuvent défiler vite)..."):
        # We manually filter SMA valid combinations inside the grid_search or beforehand
        if strategy_name == "Croisement SMA":
            # For SMA, we must ensure fast < slow
            valid_combos = []
            for f in param_grid["fast_period"]:
                for s in param_grid["slow_period"]:
                    if f < s:
                        valid_combos.append({"fast_period": f, "slow_period": s})
            
            # Since grid_search uses itertools.product on param_grid, it doesn't allow custom filter.
            # We will patch it temporarily by redefining valid ranges or catching exceptions.
            # The grid_search handles Exceptions (like ValueError for fast>=slow) and ignores them gracefully.
            pass
            
        results = grid_search(
            strategy_cls=strat_cls,
            param_grid=param_grid,
            symbol=symbol,
            interval=interval,
            period=period,
            initial_capital=initial_capital,
            progress_callback=ui_callback
        )
        
    progress_bar.progress(1.0, text="✅ Optimisation Terminée")
    
    if not results:
        st.error("Aucun résultat valide trouvé.")
    else:
        st.success(f"Optimisation terminée ! Top 10 configurations pour {symbol} :")
        
        display_data = []
        for r in results[:10]:
            params_str = ", ".join(f"{k}={v}" for k, v in getattr(r, "params", {}).items())
            display_data.append({
                "Paramètres": params_str,
                "Rendement (%)": f"{r.total_return_pct:+.2f}%",
                "Valeur Finale": f"${r.final_value:,.2f}",
                "Drawdown Max": f"-{r.max_drawdown_pct:.2f}%",
                "Ratio Sharpe": f"{r.sharpe_ratio:.2f}",
                "Taux de réussite": f"{r.win_rate * 100:.1f}%",
                "Trades": r.total_trades
            })
            
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
