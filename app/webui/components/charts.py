"""
TradeFlow — WebUI Plotly Chart Components
Reusable Plotly chart builders for candlestick, equity curve, RSI, MACD.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    trades_df: Optional[pd.DataFrame] = None,
    show_sma: bool = True,
    show_bollinger: bool = True,
) -> go.Figure:
    """
    Build a full candlestick chart with optional indicator overlays and trade markers.

    Args:
        df: OHLCV DataFrame with indicator columns (from add_all_indicators).
        symbol: Asset symbol for chart title.
        trades_df: Optional DataFrame with trade history to plot BUY/SELL markers.
        show_sma: If True, overlay SMA lines.
        show_bollinger: If True, overlay Bollinger Bands.

    Returns:
        Plotly Figure with up to 3 subplots: [Price, RSI, MACD].
    """
    # Determine how many sub-rows we need
    has_rsi = any(col.startswith("rsi_") for col in df.columns)
    has_macd = any(col.startswith("MACD_") for col in df.columns)

    subplot_rows = 1 + int(has_rsi) + int(has_macd)
    row_heights = _compute_row_heights(subplot_rows)

    subplot_titles = [f"{symbol} — Candlestick"]
    if has_rsi:
        subplot_titles.append("RSI")
    if has_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=subplot_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#00C896",
            decreasing_line_color="#FF4B6E",
        ),
        row=1,
        col=1,
    )

    # ── SMA overlays ──────────────────────────────────────────────────────────
    if show_sma:
        sma_styles = {
            "sma_20": {"color": "#FFB347", "width": 1.5, "label": "SMA 20"},
            "sma_50": {"color": "#87CEEB", "width": 1.5, "label": "SMA 50"},
            "sma_200": {"color": "#DDA0DD", "width": 1.5, "label": "SMA 200"},
        }
        for col, style in sma_styles.items():
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=style["label"],
                        line=dict(color=style["color"], width=style["width"]),
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if show_bollinger:
        bb_upper = [c for c in df.columns if c.startswith("BBU_")]
        bb_lower = [c for c in df.columns if c.startswith("BBL_")]
        bb_mid = [c for c in df.columns if c.startswith("BBM_")]

        if bb_upper and bb_lower:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[bb_upper[0]],
                    name="BB Upper",
                    line=dict(color="#9B9FFF", width=1, dash="dot"),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[bb_lower[0]],
                    name="BB Lower",
                    line=dict(color="#9B9FFF", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(155, 159, 255, 0.07)",
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )

    # ── BUY/SELL markers ─────────────────────────────────────────────────────
    if trades_df is not None and not trades_df.empty:
        _add_trade_markers(fig, df, trades_df, row=1)

    # ── RSI subplot ──────────────────────────────────────────────────────────
    current_row = 2
    if has_rsi:
        rsi_col = [c for c in df.columns if c.startswith("rsi_")][0]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[rsi_col],
                name="RSI",
                line=dict(color="#F0A500", width=1.5),
            ),
            row=current_row,
            col=1,
        )
        # Overbought / oversold bands
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,75,110,0.5)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,150,0.5)", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # ── MACD subplot ─────────────────────────────────────────────────────────
    if has_macd:
        macd_col = [c for c in df.columns if c.startswith("MACD_") and "MACDs" not in c and "MACDh" not in c]
        signal_col = [c for c in df.columns if c.startswith("MACDs_")]
        hist_col = [c for c in df.columns if c.startswith("MACDh_")]

        if macd_col and signal_col:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[macd_col[0]],
                    name="MACD",
                    line=dict(color="#00C896", width=1.5),
                ),
                row=current_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[signal_col[0]],
                    name="Signal",
                    line=dict(color="#FF4B6E", width=1.5),
                ),
                row=current_row,
                col=1,
            )

        if hist_col:
            hist_values = df[hist_col[0]]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=hist_values,
                    name="Histogram",
                    marker_color=[
                        "#00C896" if v >= 0 else "#FF4B6E" for v in hist_values.fillna(0)
                    ],
                    opacity=0.6,
                ),
                row=current_row,
                col=1,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=0, r=10, t=30, b=0),
        height=700,
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="#1E2530", showgrid=True)
    fig.update_yaxes(gridcolor="#1E2530", showgrid=True)

    return fig


def build_equity_curve_chart(
    equity_curve_df: pd.DataFrame,
    initial_capital: float,
    title: str = "Portfolio Equity Curve",
) -> go.Figure:
    """
    Build an equity curve line chart with buy-and-hold reference.

    Args:
        equity_curve_df: DataFrame with 'timestamp' and 'total_value' columns.
        initial_capital: Starting portfolio value for baseline reference.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_curve_df["timestamp"],
            y=equity_curve_df["total_value"],
            name="Portfolio Value",
            line=dict(color="#00C896", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 200, 150, 0.08)",
        )
    )

    # Flat reference line at initial capital
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Initial Capital",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=350,
        margin=dict(l=0, r=10, t=50, b=0),
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="#1E2530")
    fig.update_yaxes(gridcolor="#1E2530")

    return fig


def build_returns_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """
    Build a histogram of trade P&L distribution.

    Args:
        trades_df: DataFrame with 'pnl' column (SELL trades only).

    Returns:
        Plotly Figure.
    """
    sell_trades = trades_df[trades_df["side"] == "SELL"] if not trades_df.empty else trades_df

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sell_trades["pnl"] if not sell_trades.empty else [],
            nbinsx=20,
            name="Trade P&L",
            marker_color="#00C896",
            opacity=0.8,
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        title="Trade P&L Distribution",
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        xaxis_title="P&L",
        yaxis_title="Count",
        height=300,
        margin=dict(l=0, r=10, t=50, b=0),
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="#1E2530")
    fig.update_yaxes(gridcolor="#1E2530")

    return fig


# ─── Private helpers ──────────────────────────────────────────────────────────

def _add_trade_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    row: int,
) -> None:
    """Overlay BUY and SELL markers on the price chart with reason tooltips."""
    buy_trades = trades_df[trades_df["side"] == "BUY"]
    sell_trades = trades_df[trades_df["side"] == "SELL"]

    reason_col = "reason" if "reason" in trades_df.columns else None

    if not buy_trades.empty:
        hover_texts = (
            ["🟢 ACHAT<br>" + str(r) if r else "🟢 ACHAT" for r in buy_trades[reason_col]]
            if reason_col else ["🟢 ACHAT"] * len(buy_trades)
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(buy_trades["timestamp"]),
                y=buy_trades["price"],
                mode="markers",
                name="BUY",
                text=hover_texts,
                hovertemplate="<b>%{text}</b><br>Prix: $%{y:.2f}<br>Date: %{x}<extra></extra>",
                marker=dict(symbol="triangle-up", size=14, color="#00C896", line=dict(width=1, color="#fff")),
            ),
            row=row,
            col=1,
        )

    if not sell_trades.empty:
        hover_texts = (
            ["🔴 VENTE<br>" + str(r) if r else "🔴 VENTE" for r in sell_trades[reason_col]]
            if reason_col else ["🔴 VENTE"] * len(sell_trades)
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sell_trades["timestamp"]),
                y=sell_trades["price"],
                mode="markers",
                name="SELL",
                text=hover_texts,
                hovertemplate="<b>%{text}</b><br>Prix: $%{y:.2f}<br>Date: %{x}<extra></extra>",
                marker=dict(symbol="triangle-down", size=14, color="#FF4B6E", line=dict(width=1, color="#fff")),
            ),
            row=row,
            col=1,
        )


def _compute_row_heights(n_rows: int) -> list[float]:
    """Compute proportional row heights for subplots."""
    if n_rows == 1:
        return [1.0]
    if n_rows == 2:
        return [0.65, 0.35]
    return [0.55, 0.22, 0.23]
