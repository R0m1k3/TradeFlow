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

# ── Global CSS injection ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root dark theme */
    :root {
        --bg-primary: #0D1117;
        --bg-secondary: #161B22;
        --bg-card: #1C2333;
        --accent-green: #00C896;
        --accent-red: #FF4B6E;
        --accent-blue: #58A6FF;
        --accent-purple: #BC8CFF;
        --text-primary: #E6EDF3;
        --text-secondary: #8B949E;
        --border: #30363D;
        --border-glow: rgba(0, 200, 150, 0.25);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1600px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-green);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--accent-green);
        box-shadow: 0 0 12px var(--border-glow);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* Positive delta green, negative red */
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00C896, #00A87A);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 200, 150, 0.4);
    }

    /* Select boxes, inputs */
    .stSelectbox > div[data-baseweb="select"] > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: var(--bg-secondary) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--accent-green) !important;
    }

    /* Dividers */
    hr { border-color: var(--border); }

    /* Remove Streamlit footer/branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Custom card container */
    .tf-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .tf-badge-green {
        display: inline-block;
        background: rgba(0,200,150,0.15);
        color: var(--accent-green);
        border: 1px solid rgba(0,200,150,0.3);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tf-badge-red {
        display: inline-block;
        background: rgba(255,75,110,0.15);
        color: var(--accent-red);
        border: 1px solid rgba(255,75,110,0.3);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
