"""
TradeFlow — Portfolio Page
Vue détaillée du portefeuille : courbe d'évolution, positions ouvertes,
historique complet des trades et statistiques par action.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.bot.live_trader import get_active_live_session
from app.data.fetcher import fetch_ohlcv
from app.data.nasdaq import get_display_name, get_currency, format_price, format_price_sign
from app.database.models import Portfolio as PortfolioModel, Trade
from app.database.session import get_session, init_database
from app.webui.explanations import pnl_color, pnl_class

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradeFlow — Portefeuille",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_css_path = Path(__file__).resolve().parents[1] / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}</style>", unsafe_allow_html=True)

st_autorefresh(interval=60_000, key="portfolio_refresh")

if "db_initialized" not in st.session_state:
    init_database()
    st.session_state.db_initialized = True

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="tf-header">'
    '<div style="display:flex;align-items:center;">'
    '<span class="tf-brand">TradeFlow</span>'
    '<span class="tf-brand-sub">Portefeuille</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _load_equity(run_id: int) -> pd.DataFrame:
    s = get_session()
    try:
        rows = (
            s.query(PortfolioModel)
            .filter_by(sim_run_id=run_id)
            .order_by(PortfolioModel.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": r.timestamp, "total_value": r.total_value, "cash": r.cash}
            for r in rows
        ])
    finally:
        s.close()


@st.cache_data(ttl=60, show_spinner=False)
def _load_positions(run_id: int) -> dict:
    s = get_session()
    try:
        snap = (
            s.query(PortfolioModel)
            .filter_by(sim_run_id=run_id)
            .order_by(PortfolioModel.timestamp.desc())
            .first()
        )
        if snap:
            return {
                "cash": snap.cash,
                "total_value": snap.total_value,
                "positions": json.loads(snap.positions_json),
            }
        return {}
    finally:
        s.close()


@st.cache_data(ttl=60, show_spinner=False)
def _load_trades(run_id: int) -> pd.DataFrame:
    s = get_session()
    try:
        trades = (
            s.query(Trade)
            .filter_by(sim_run_id=run_id)
            .order_by(Trade.timestamp.desc())
            .all()
        )
        return pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    finally:
        s.close()


@st.cache_data(ttl=300, show_spinner=False)
def _sparkline(sym: str) -> list[float]:
    try:
        df = fetch_ohlcv(sym, interval="1h", period="5d")
        if df is not None and not df.empty:
            return df["close"].tolist()[-40:]
    except Exception:
        pass
    return []


@st.cache_data(ttl=300, show_spinner=False)
def _current_price(sym: str) -> float | None:
    try:
        df = fetch_ohlcv(sym, interval="1h", period="2d")
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
    except Exception:
        pass
    return None


def _sparkline_svg(prices: list[float], color: str, width: int = 120, height: int = 40) -> str:
    if len(prices) < 2:
        return ""
    mn, mx = min(prices), max(prices)
    rng = mx - mn or 1
    pts = []
    for i, p in enumerate(prices):
        x = i / (len(prices) - 1) * width
        y = height - (p - mn) / rng * height
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ── Load active session ────────────────────────────────────────────────────────

active = get_active_live_session()

if active is None:
    st.markdown(
        '<div class="tf-empty" style="margin-top:4rem;">'
        '<div style="font-size:2rem;margin-bottom:1rem;">📊</div>'
        '<div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;">Aucune session active</div>'
        '<div class="tf-explain">Demarrez le bot depuis la page principale pour voir votre portefeuille.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

run_id = active["id"]
initial_cap = active["initial_capital"]

equity_df = _load_equity(run_id)
latest = _load_positions(run_id)
trades_df = _load_trades(run_id)

cash = latest.get("cash", initial_cap)
total_value = latest.get("total_value", initial_cap)
positions = latest.get("positions", {})
pnl = total_value - initial_cap
pnl_pct = pnl / initial_cap * 100 if initial_cap else 0.0

sells = trades_df[trades_df["side"] == "SELL"] if not trades_df.empty else pd.DataFrame()
n_wins = int((sells["pnl"] > 0).sum()) if not sells.empty else 0
n_sells = len(sells)
win_rate = n_wins / n_sells * 100 if n_sells else 0.0
total_realized = float(sells["pnl"].sum()) if not sells.empty else 0.0

# ── Top metrics ────────────────────────────────────────────────────────────────

pnl_c = pnl_color(pnl)
start = active.get("start_date", "—")
st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:1.5rem;">'
    f'<div class="tf-metric"><div class="tf-metric-value" style="color:{pnl_c}">{total_value:,.2f} €</div><div class="tf-metric-label">Valeur totale</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value" style="color:{pnl_c}">{pnl_pct:+.2f}%</div><div class="tf-metric-label">Rendement</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value" style="color:{pnl_color(total_realized)}">{total_realized:+,.2f} €</div><div class="tf-metric-label">P&L réalisé</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value">{cash:,.2f} €</div><div class="tf-metric-label">Cash dispo</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value">{win_rate:.0f}%</div><div class="tf-metric-label">Win Rate ({n_wins}/{n_sells})</div></div>'
    f'<div class="tf-metric"><div class="tf-metric-value">{len(trades_df) if not trades_df.empty else 0}</div><div class="tf-metric-label">Trades totaux</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Equity curve ───────────────────────────────────────────────────────────────

st.markdown('<div class="tf-section">Evolution du portefeuille</div>', unsafe_allow_html=True)

if not equity_df.empty:
    color = "#00C896" if total_value >= initial_cap else "#FF4B6E"
    fill_color = "rgba(0,200,150,0.07)" if color == "#00C896" else "rgba(255,75,110,0.07)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"], y=equity_df["total_value"],
        mode="lines", line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=fill_color, name="Valeur totale",
        hovertemplate="%{x|%d/%m %H:%M}<br><b>%{y:,.2f} €</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"], y=equity_df["cash"],
        mode="lines", line=dict(color="#58A6FF", width=1.2, dash="dot"),
        name="Cash", opacity=0.6,
        hovertemplate="%{x|%d/%m %H:%M}<br>Cash: %{y:,.2f} €<extra></extra>",
    ))
    fig.add_hline(
        y=initial_cap, line_dash="dash", line_color="rgba(255,255,255,0.2)",
        annotation_text=f"Capital initial {initial_cap:,.0f} €",
        annotation_font_color="#8B949E", annotation_font_size=11,
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111418", plot_bgcolor="#111418",
        height=320, margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#8B949E")),
        font=dict(family="Inter,sans-serif", size=12, color="#8B949E"),
    )
    fig.update_xaxes(gridcolor="#1E2530", color="#8B949E")
    fig.update_yaxes(gridcolor="#1E2530", color="#8B949E", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(
        '<div class="tf-empty">En attente des premieres donnees du bot...</div>',
        unsafe_allow_html=True,
    )

# ── Positions ouvertes ─────────────────────────────────────────────────────────

st.markdown('<div class="tf-section">Positions ouvertes</div>', unsafe_allow_html=True)

if positions:
    with st.spinner("Chargement des positions..."):
        # Fetch current prices in parallel
        import concurrent.futures
        syms = list(positions.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            price_futures = {sym: pool.submit(_current_price, sym) for sym in syms}
            spark_futures = {sym: pool.submit(_sparkline, sym) for sym in syms}
            cur_prices = {sym: price_futures[sym].result() for sym in syms}
            sparklines = {sym: spark_futures[sym].result() for sym in syms}

    pos_rows = []
    for sym, pos in positions.items():
        qty = pos.get("qty", 0)
        avg = pos.get("avg_price", 0)
        cur = cur_prices.get(sym) or pos.get("current_price", avg)
        unreal = (cur - avg) * qty
        unreal_pct = (cur - avg) / avg * 100 if avg else 0.0
        currency = get_currency(sym)
        spark_prices = sparklines.get(sym, [])
        spark_color = "#00C896" if unreal >= 0 else "#FF4B6E"
        spark_html = _sparkline_svg(spark_prices, spark_color)
        pos_rows.append({
            "sym": sym,
            "name": get_display_name(sym),
            "qty": qty,
            "avg": avg,
            "cur": cur,
            "unreal": unreal,
            "unreal_pct": unreal_pct,
            "currency": currency,
            "spark": spark_html,
            "pnl_cls": pnl_class(unreal),
            "pnl_c": pnl_color(unreal),
        })

    # Render positions table
    rows_html = ""
    for r in sorted(pos_rows, key=lambda x: abs(x["unreal"]), reverse=True):
        border = "#00C896" if r["unreal"] >= 0 else "#FF4B6E"
        rows_html += (
            f'<tr style="border-left:3px solid {border};">'
            f'<td style="font-weight:700;font-family:monospace;padding:0.7rem 1rem;">{r["sym"]}</td>'
            f'<td style="color:#8B949E;padding:0.7rem 0.5rem;">{r["name"]}</td>'
            f'<td style="font-family:monospace;padding:0.7rem 0.5rem;">{r["qty"]:.4f}</td>'
            f'<td style="font-family:monospace;padding:0.7rem 0.5rem;">{format_price(r["avg"], r["currency"])}</td>'
            f'<td style="font-family:monospace;padding:0.7rem 0.5rem;">{format_price(r["cur"], r["currency"])}</td>'
            f'<td style="font-weight:700;color:{r["pnl_c"]};font-family:monospace;padding:0.7rem 0.5rem;">'
            f'{format_price_sign(r["unreal"], r["currency"])}</td>'
            f'<td style="font-weight:700;color:{r["pnl_c"]};font-family:monospace;padding:0.7rem 0.5rem;">'
            f'{r["unreal_pct"]:+.2f}%</td>'
            f'<td style="padding:0.7rem 1rem;">{r["spark"]}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div class="tf-table-wrapper">'
        f'<table class="tf-stock-table">'
        f'<thead><tr>'
        f'<th>Symbole</th><th>Société</th><th>Quantité</th>'
        f'<th>Prix moyen</th><th>Prix actuel</th>'
        f'<th>P&L non réalisé</th><th>%</th><th>Évolution 5j</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="tf-empty">Aucune position ouverte actuellement.</div>',
        unsafe_allow_html=True,
    )

# ── Par action — P&L réalisé ───────────────────────────────────────────────────

if not trades_df.empty and not sells.empty:
    st.markdown('<div class="tf-section">P&L réalisé par action</div>', unsafe_allow_html=True)

    by_sym = sells.groupby("symbol").agg(
        pnl_total=("pnl", "sum"),
        n_trades=("pnl", "count"),
        n_wins=("pnl", lambda x: (x > 0).sum()),
    ).reset_index().sort_values("pnl_total", ascending=False)

    colors = ["#00C896" if v >= 0 else "#FF4B6E" for v in by_sym["pnl_total"]]
    fig2 = go.Figure(go.Bar(
        x=by_sym["symbol"], y=by_sym["pnl_total"],
        marker_color=colors,
        text=[f"{v:+.2f} €" for v in by_sym["pnl_total"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P&L: %{y:+,.2f} €<extra></extra>",
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#111418", plot_bgcolor="#111418",
        height=260, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
        font=dict(family="Inter,sans-serif", size=11, color="#8B949E"),
        yaxis=dict(gridcolor="#1E2530", color="#8B949E", tickformat="+,.0f"),
        xaxis=dict(color="#8B949E"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Historique complet des trades ──────────────────────────────────────────────

st.markdown('<div class="tf-section">Historique des trades</div>', unsafe_allow_html=True)

if trades_df.empty:
    st.markdown(
        '<div class="tf-empty">Aucun trade exécuté pour cette session.</div>',
        unsafe_allow_html=True,
    )
else:
    col_filter, col_count = st.columns([3, 1])
    with col_filter:
        side_filter = st.radio(
            "Afficher",
            ["Tous", "Achats", "Ventes"],
            horizontal=True,
            key="pf_side_filter",
        )
    with col_count:
        st.markdown(
            f'<div style="text-align:right;color:#8B949E;font-size:0.82rem;padding-top:0.5rem;">'
            f'{len(trades_df)} trades au total</div>',
            unsafe_allow_html=True,
        )

    if side_filter == "Achats":
        display_df = trades_df[trades_df["side"] == "BUY"]
    elif side_filter == "Ventes":
        display_df = trades_df[trades_df["side"] == "SELL"]
    else:
        display_df = trades_df

    rows_html = ""
    for _, row in display_df.head(100).iterrows():
        ts = pd.to_datetime(row.get("timestamp", ""))
        t_str = ts.strftime("%d/%m %H:%M") if pd.notna(ts) else "—"
        side = row.get("side", "")
        sym = row.get("symbol", "")
        qty = row.get("quantity", 0)
        price_val = row.get("price", 0)
        currency = get_currency(sym)
        pnl_val = row.get("pnl", 0)
        reason = (row.get("reason", "") or "")[:60]

        side_color = "#00C896" if side == "BUY" else "#FF4B6E"
        side_label = "ACHAT" if side == "BUY" else "VENTE"

        pnl_html = ""
        if side == "SELL" and pd.notna(pnl_val):
            c = pnl_color(float(pnl_val))
            pnl_html = f'<span style="color:{c};font-weight:700;">{format_price_sign(float(pnl_val), currency)}</span>'

        rows_html += (
            f'<tr>'
            f'<td class="tf-trade-time" style="padding:0.55rem 0.8rem;white-space:nowrap;">{t_str}</td>'
            f'<td style="padding:0.55rem 0.5rem;"><span style="color:{side_color};font-weight:700;font-size:0.78rem;">{side_label}</span></td>'
            f'<td style="font-weight:700;font-family:monospace;padding:0.55rem 0.5rem;">{sym}</td>'
            f'<td style="color:#8B949E;padding:0.55rem 0.5rem;">{get_display_name(sym)}</td>'
            f'<td style="font-family:monospace;padding:0.55rem 0.5rem;color:#8B949E;">{qty:.4f}</td>'
            f'<td style="font-family:monospace;padding:0.55rem 0.5rem;">{format_price(float(price_val), currency)}</td>'
            f'<td style="padding:0.55rem 0.5rem;">{pnl_html}</td>'
            f'<td style="color:#8B949E;font-size:0.75rem;padding:0.55rem 0.8rem;">{reason}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div class="tf-table-wrapper">'
        f'<table class="tf-stock-table">'
        f'<thead><tr>'
        f'<th>Date</th><th>Type</th><th>Symbole</th><th>Société</th>'
        f'<th>Quantité</th><th>Prix</th><th>P&L</th><th>Raison</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    if len(display_df) > 100:
        st.caption(f"Affichage limité aux 100 premiers trades sur {len(display_df)}.")

# ── Footer ──────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="tf-footer">Session #{run_id} · Depuis le {start} · '
    f'Dernière MàJ : {datetime.now().strftime("%H:%M:%S")}</div>',
    unsafe_allow_html=True,
)
