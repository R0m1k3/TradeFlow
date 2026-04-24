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
    "<style>html,body,[class*='css']{font-family:'Inter',sans-serif!important;}"
    ".main .block-container{padding:1.5rem 2rem;max-width:1600px;}"
    "[data-testid='stMetric']{background:#1C2333;border:1px solid #30363D;border-radius:12px;padding:1rem 1.25rem;}"
    "#MainMenu,footer,header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

init_database()

# ── Constants ─────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict[str, type] = {
    "SMA Crossover (20/50)": SmaCrossoverStrategy,
    "RSI (14) [30/70]": RsiStrategy,
    "MACD (12/26/9)": MacdStrategy,
}

DEFAULT_SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "MC.PA", "TTE.PA"]
INTERVALS = {"1 Hour": "1h", "Daily": "1d", "30 Minutes": "30m", "15 Minutes": "15m"}
PERIODS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
}

st.markdown("## 🚀 Simulation")
st.markdown("---")

# ── Configuration form ────────────────────────────────────────────────────────
with st.form("simulation_form", clear_on_submit=False):
    st.markdown("### Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox("Asset Symbol", DEFAULT_SYMBOLS, key="sim_symbol")
        custom_sym = st.text_input("Custom ticker (overrides above)", placeholder="e.g. NVDA", key="sim_custom")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()), key="sim_strategy")

    with col2:
        interval_label = st.selectbox("Bar Interval", list(INTERVALS.keys()), key="sim_interval")
        period_label = st.selectbox("Historical Period", list(PERIODS.keys()), index=1, key="sim_period")
        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=1000, max_value=10_000_000,
            value=10_000, step=1000, key="sim_capital"
        )

    with col3:
        st.markdown("**Broker Settings**")
        commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01, key="sim_commission") / 100
        slippage = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.005, key="sim_slippage") / 100
        position_size = st.slider("Position Size (%)", 10, 100, 95, 5, key="sim_pos_size") / 100

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-strategy comparison
    st.markdown("### Multi-Strategy Comparison *(optional)*")
    run_comparison = st.checkbox(
        "Run all 3 strategies on the same symbol and compare",
        value=False,
        key="sim_compare",
    )

    submitted = st.form_submit_button("🚀 Launch Simulation", use_container_width=True)

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
        st.markdown(f"---\n#### Running: **{strat_name}** on `{effective_symbol}`")

        progress_bar = st.progress(0, text=f"Initializing {strat_name}…")
        status_placeholder = st.empty()

        def _update_progress(pct: float) -> None:
            progress_bar.progress(min(pct, 1.0), text=f"Processing bars… {pct*100:.0f}%")

        # Instantiate strategy with default params
        strategy = strat_cls()

        with st.spinner(f"Fetching data and running backtest…"):
            result = engine.run(
                strategy=strategy,
                symbol=effective_symbol,
                interval=interval,
                period=period,
                initial_capital=initial_capital,
                progress_callback=_update_progress,
            )

        progress_bar.progress(1.0, text="✅ Complete")

        if result is None:
            st.error(
                f"❌ Simulation failed for **{strat_name}** on **{effective_symbol}**. "
                "Check the symbol and interval."
            )
            continue

        results.append(result)

        # ── Results display ────────────────────────────────────────────────
        ret_color = "#00C896" if result.total_return_pct >= 0 else "#FF4B6E"
        st.markdown(
            f"""
            <div style="background:#1C2333;border:1px solid #30363D;border-radius:12px;padding:1.25rem;margin:0.5rem 0;">
                <div style="font-weight:600;font-size:1rem;margin-bottom:0.75rem;color:#E6EDF3;">
                    {strat_name} Results
                </div>
                <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.75rem;text-align:center;">
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Final Value</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">${result.final_value:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Return</div>
                        <div style="font-size:1.1rem;font-weight:600;color:{ret_color};">{result.total_return_pct:+.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Sharpe</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">{result.sharpe_ratio:.2f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Max DD</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#FF4B6E;">-{result.max_drawdown_pct:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Win Rate</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">{result.win_rate * 100:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;">Trades</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">{result.total_trades}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Equity curve for this run
        if result.equity_curve:
            equity_df = pd.DataFrame(result.equity_curve, columns=["timestamp", "total_value"])
            fig = build_equity_curve_chart(equity_df, initial_capital, title=f"Equity Curve — {strat_name}")
            st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    if run_comparison and len(results) > 1:
        st.markdown("---")
        st.markdown("### 🏆 Strategy Comparison")
        comp_data = [
            {
                "Strategy": r.strategy_name,
                "Return (%)": f"{r.total_return_pct:+.2f}%",
                "Final Value": f"${r.final_value:,.2f}",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "Max Drawdown": f"-{r.max_drawdown_pct:.2f}%",
                "Win Rate": f"{r.win_rate * 100:.1f}%",
                "Trades": r.total_trades,
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    if results:
        st.success(f"✅ {len(results)} simulation(s) completed and saved to database.")
        st.info("📊 View results in the **Dashboard** and **Trades** pages.")
