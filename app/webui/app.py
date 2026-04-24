"""
TradeFlow — Streamlit WebUI Entry Point
Multi-page navigation app with premium dark theme.
"""

import sys
import os

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="TradeFlow",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "TradeFlow — Simulateur de Trading Algorithmique",
    },
)

# L'injection CSS personnalisée a été retirée.
# Le thème est désormais géré proprement par .streamlit/config.toml
# pour garantir une lisibilité parfaite sur tous les appareils.

# ── Sidebar brand ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-size: 2.2rem;">📈</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #00C896; letter-spacing: 0.05em;">
                TradeFlow
            </div>
            <div style="font-size: 0.72rem; color: #8B949E; margin-top: 2px;">
                Simulateur de Trading Algorithmique
            </div>
        </div>
        <hr style="border-color: #30363D; margin: 0 0 1rem;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Navigation")

# ── Home page content ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 3rem 0 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📈</div>
        <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #00C896, #58A6FF);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            TradeFlow
        </h1>
        <p style="color: #8B949E; font-size: 1.1rem; margin-top: 0.5rem;">
            Simulateur de Trading Algorithmique — Marchés US &amp; Européens
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    ("📊", "Tableau de Bord", "Aperçu du portefeuille, courbe de capital et historique des simulations."),
    ("🕯️", "Graphiques", "Graphiques en chandeliers interactifs avec indicateurs et marqueurs de trades."),
    ("📋", "Transactions", "Historique complet des transactions avec statistiques et analyse P&L."),
    ("🚀", "Simulation", "Configurez et lancez des backtests sur des actifs US et Européens."),
    ("⚙️", "Optimisation", "Trouvez automatiquement les meilleurs paramètres (Grid Search) pour une stratégie."),
]

for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], cards):
    with col:
        st.markdown(
            f"""
            <div class="tf-card" style="text-align:center; cursor:pointer;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600; font-size:1rem; color:#E6EDF3; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.8rem; color:#8B949E; line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 **Utilisez le menu latéral pour naviguer entre les pages.** Commencez par **Simulation** pour lancer votre premier backtest.")
