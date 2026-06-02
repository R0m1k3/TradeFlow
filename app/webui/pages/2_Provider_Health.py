"""
TradeFlow — Provider Health Dashboard

Visualise the state of every data provider in real time:
* Which providers are configured (API key present)
* Per-provider circuit breaker state (closed / open / half_open)
* Per-provider negative cache counters (how many tickers are in skip mode)
* Per-provider adaptive backoff (how many tickers are blocked)
* Coverage matrix (which markets each provider supports)
* Live tester: pick a ticker, click "Test chain", see which source answered
  and how long it took — plus a per-source breakdown of why each failed.

This is the dashboard for the multi-source data layer (Finnhub, Twelve Data,
Alpha Vantage, Yahoo). It pairs with the /api/admin/providers and
/api/admin/test-source endpoints in app/webui/server.py.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 10s so the dashboard reflects live provider state
st_autorefresh(interval=10000, key="provider_health_refresh")

st.set_page_config(
    page_title="TradeFlow — Provider Health",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_css_path = Path(__file__).resolve().parents[1] / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .tf-card {
    background: var(--bg-tint, #0e1117);
    border: 1px solid var(--hairline, #1e2128);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .tf-state-ok { color: #00C896; font-weight: 700; }
  .tf-state-warn { color: #FFD43B; font-weight: 700; }
  .tf-state-bad { color: #FF4B6E; font-weight: 700; }
  .tf-state-na { color: #8B949E; font-weight: 700; }
  .tf-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .tf-pill-on { background: rgba(0, 200, 150, 0.15); color: #00C896; }
  .tf-pill-off { background: rgba(255, 75, 110, 0.15); color: #FF4B6E; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Provider Health")
st.caption("État temps réel des sources de données. Auto-refresh 10s.")

# ── 1. Provider cards ──────────────────────────────────────────────────────

try:
    import requests
    providers_resp = requests.get("http://localhost:8501/api/admin/providers", timeout=5)
    providers_data = providers_resp.json()
except Exception as exc:
    st.error(f"Impossible de joindre l'API: {exc}")
    st.stop()

providers = providers_data.get("providers", [])
priority = providers_data.get("default_priority", [])

st.subheader("Sources configurées")

# Summary stats
cols = st.columns(4)
for i, p in enumerate(providers):
    with cols[i]:
        name = p["name"].replace("_", " ").title()
        available = p["available"]
        cls = "tf-pill-on" if available else "tf-pill-off"
        state_cls = "tf-state-ok" if available else "tf-state-na"
        resilience = p["resilience"]
        breaker_state = resilience["breaker"]["state"]

        st.markdown(f"""
        <div class="tf-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0;">{name}</h3>
            <span class="tf-pill {cls}">{("✓ actif" if available else "✗ inactif")}</span>
          </div>
          <div style="margin-top:8px; font-size:0.85rem; color:#8B949E;">
            Breaker: <span class="{('tf-state-ok' if breaker_state == 'closed' else 'tf-state-bad' if breaker_state == 'open' else 'tf-state-warn')}">{breaker_state.upper()}</span>
            · {resilience['breaker']['failure_rate']*100:.0f}% fail
          </div>
          <div style="margin-top:6px; font-size:0.8rem; color:#8B949E;">
            Skip: {resilience['negative_cache']['skipping']} · Degraded: {resilience['negative_cache']['degraded']} · Backoff: {resilience['backoff']['blocked']}
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── 2. Coverage matrix ─────────────────────────────────────────────────────

st.subheader("Matrice de couverture")
markets = ["US", "EU", "UK", "CH", "DE", "FR", "NL", "AS", "CRYPTO", "FX"]
cov_data = []
for p in providers:
    row = {"Provider": p["name"].replace("_", " ").title()}
    for m in markets:
        row[m] = "✓" if m in p["coverage"]["markets"] else ""
    row["Intraday"] = "✓" if p["coverage"]["intraday"] else ""
    row["Fondamentaux"] = "✓" if p["coverage"].get("has_fundamentals") else ""
    cov_data.append(row)
df_cov = pd.DataFrame(cov_data).set_index("Provider")
st.dataframe(df_cov, use_container_width=True)

# ── 3. Live tester ─────────────────────────────────────────────────────────

st.subheader("Test live d'une chaîne")
st.caption("Lance `SourceRouter.fetch_ohlcv()` pour un ticker donné et affiche la source qui a répondu.")

# Settings: input + save provider keys to the DB-backed SettingsStore
with st.expander("🔑 Clés API (stockées dans `data/settings.json` — persistent)", expanded=False):
    st.caption("Les clés sont sauvegardées via `/api/config` et persistent après restart. Une fois entrées, l'app les utilise automatiquement à chaque appel.")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        k_finnhub = st.text_input("Finnhub", value="", type="password",
                                   key="k_finnhub", placeholder="d7lu8...")
    with col_k2:
        k_twelve = st.text_input("Twelve Data", value="", type="password",
                                  key="k_twelve", placeholder="xxx")
    with col_k3:
        k_av = st.text_input("Alpha Vantage", value="", type="password",
                              key="k_av", placeholder="yyy")

    if st.button("💾 Sauvegarder les clés", key="save_keys"):
        body = {
            "data_finnhub_key": k_finnhub or "",
            "data_twelve_data_key": k_twelve or "",
            "data_alpha_vantage_key": k_av or "",
        }
        try:
            r = requests.post("http://localhost:8501/api/config", json=body, timeout=5)
            r.raise_for_status()
            st.success("Clés sauvegardées. Recharge la page pour voir l'effet.")
        except Exception as exc:
            st.error(f"Erreur: {exc}")

col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    test_symbol = st.text_input("Symbole", value="ROG.SW", placeholder="ex: AAPL, ROG.SW, MC.PA, NESN.SW")
with col2:
    test_interval = st.selectbox("Intervalle", ["1m", "5m", "15m", "1h", "1d"], index=4)
with col3:
    test_period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.button("🚀 Tester la chaîne", type="primary"):
    if not test_symbol:
        st.error("Entrez un symbole.")
    else:
        with st.spinner(f"Test {test_symbol} en cours..."):
            try:
                resp = requests.post(
                    "http://localhost:8501/api/admin/test-source",
                    params={"symbol": test_symbol, "interval": test_interval, "period": test_period},
                    timeout=30,
                )
                data = resp.json()
            except Exception as exc:
                st.error(f"Erreur API: {exc}")
                data = None

        if data:
            chosen = data.get("chosen")
            attempts = data.get("attempts", [])
            if chosen:
                st.success(f"✅ **{chosen}** a répondu avec **{data.get('bars_returned', 0)} bougies** pour {test_symbol}")
            else:
                st.error(f"❌ Aucune source n'a pu fournir {test_symbol}. Détail ci-dessous.")

            rows = []
            for a in attempts:
                rows.append({
                    "Source": a["source"].replace("_", " ").title(),
                    "État": (
                        "✓ succès" if a.get("succeeded") else
                        "↷ skip" if a.get("skipped_reason") else
                        "✗ erreur"
                    ),
                    "Latence (ms)": a.get("latency_ms", 0),
                    "Détail": (
                        f"{a.get('bars', 0)} bars" if a.get("succeeded") else
                        a.get("skipped_reason", "") or
                        f"{a.get('error_kind', '?')}: {a.get('error', '')[:80]}"
                    ),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 4. Per-ticker source tracking ──────────────────────────────────────────

st.subheader("Dernières sources par ticker (test rapide)")
st.caption("Teste une série de tickers représentatifs et montre qui a répondu.")

default_test_tickers = [
    "AAPL", "MSFT", "NVDA",          # US
    "MC.PA", "PHIA.AS",              # EU Paris + Amsterdam
    "ROG.SW", "NESN.SW", "NOVN.SW",  # Swiss (Yahoo 404)
    "VOD.L", "TSCO.L",                # London
    "SAP.DE",                        # Xetra
]

if st.button("🔄 Lancer le balayage"):
    rows = []
    progress = st.progress(0.0, text="Démarrage...")
    for i, sym in enumerate(default_test_tickers):
        try:
            r = requests.post(
                "http://localhost:8501/api/admin/test-source",
                params={"symbol": sym, "interval": "1d", "period": "1mo"},
                timeout=20,
            )
            data = r.json()
            chosen = data.get("chosen") or "—"
            attempts = data.get("attempts", [])
            first_success = next((a for a in attempts if a.get("succeeded")), None)
            bars = first_success.get("bars", 0) if first_success else 0
            latency = first_success.get("latency_ms", 0) if first_success else 0
            rows.append({
                "Ticker": sym,
                "Source": chosen.replace("_", " ").title() if chosen else "—",
                "Bars": bars,
                "Latence (ms)": latency,
                "Marché": next((m for m in ["US", "FR", "NL", "CH", "UK", "DE"]
                              if (sym.endswith(".PA") and m == "FR")
                              or (sym.endswith(".AS") and m == "NL")
                              or (sym.endswith(".SW") and m == "CH")
                              or (sym.endswith(".L") and m == "UK")
                              or (sym.endswith(".DE") and m == "DE")
                              or (not any(sym.endswith(s) for s in [".PA", ".AS", ".SW", ".L", ".DE"]) and m == "US")), "?"),
            })
        except Exception as exc:
            rows.append({"Ticker": sym, "Source": f"erreur: {exc}", "Bars": 0, "Latence (ms)": 0, "Marché": "?"})
        progress.progress((i + 1) / len(default_test_tickers), text=f"Test {sym}...")
    progress.empty()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 5. Resilience detail ───────────────────────────────────────────────────

st.subheader("Détail de la couche de résilience")
try:
    r = requests.get("http://localhost:8501/api/admin/resilience", timeout=5)
    res_data = r.json().get("guards", [])
    if res_data:
        rows = []
        for g in res_data:
            br = g["breaker"]
            nc = g["negative_cache"]
            bo = g["backoff"]
            rows.append({
                "Source": g["name"].replace("_", " ").title(),
                "Breaker": br["state"].upper(),
                "Failure rate": f"{br['failure_rate']*100:.0f}%",
                "Calls/30s": br["calls_in_window"],
                "Skip (NC)": nc["skipping"],
                "Degraded (NC)": nc["degraded"],
                "Backoff blocked": bo["blocked"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"Impossible de charger /api/admin/resilience: {exc}")
