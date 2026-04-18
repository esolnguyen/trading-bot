"""Render the Trades and Portfolio tabs."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.legacy.dashboard.loaders import (
    load_closed_trades,
    load_statistics,
    load_trade_history,
    load_trades,
)


def render_trades(symbol: str) -> None:
    df_hist = load_trade_history(symbol)
    df_trades = load_trades()

    pnl_col = None
    df_for_chart = pd.DataFrame()
    ts_col: str | None = None
    if not df_hist.empty:
        for col in ("pnl_pct", "pnl", "realized_pnl_pct"):
            if col in df_hist.columns:
                pnl_col = col
                break
    if pnl_col and not df_hist.empty:
        ts_col = next(
            (c for c in ("timestamp", "close_time", "time") if c in df_hist.columns),
            None,
        )
        if ts_col:
            df_for_chart = (
                df_hist[[ts_col, pnl_col]].dropna().sort_values(ts_col).copy()
            )
            df_for_chart["cumulative_pnl"] = pd.to_numeric(
                df_for_chart[pnl_col], errors="coerce"
            ).cumsum()
            df_for_chart["win"] = df_for_chart[pnl_col].apply(lambda v: float(v) >= 0)

    if not df_for_chart.empty and ts_col is not None and pnl_col is not None:
        _render_cumulative_pnl(df_for_chart, ts_col, pnl_col)

    if not df_hist.empty:
        _render_trade_history_table(df_hist)
    elif not df_trades.empty:
        _render_executed_trades(df_trades)
    else:
        st.info("No trade records found yet.")


def _render_cumulative_pnl(df: pd.DataFrame, ts_col: str, pnl_col: str) -> None:
    st.subheader("Cumulative P&L")
    area_chart = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#58a6ff", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#58a6ff", offset=0),
                    alt.GradientStop(color="rgba(88,166,255,0)", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X(f"{ts_col}:T", title="Time"),
            y=alt.Y("cumulative_pnl:Q", title="Cumulative P&L (%)"),
            tooltip=[
                f"{ts_col}:T",
                alt.Tooltip("cumulative_pnl:Q", format=".3f"),
                alt.Tooltip(f"{pnl_col}:Q", format=".3f", title="Trade P&L"),
            ],
        )
        .properties(height=260)
    )

    points = (
        alt.Chart(df)
        .mark_point(size=60, filled=True)
        .encode(
            x=alt.X(f"{ts_col}:T"),
            y=alt.Y("cumulative_pnl:Q"),
            color=alt.Color(
                "win:N",
                scale=alt.Scale(domain=[True, False], range=["#3fb950", "#f85149"]),
                legend=None,
            ),
            tooltip=[f"{ts_col}:T", alt.Tooltip("cumulative_pnl:Q", format=".3f")],
        )
    )
    st.altair_chart((area_chart + points).interactive(), use_container_width=True)


def _render_trade_history_table(df_hist: pd.DataFrame) -> None:
    st.subheader("Trade history")
    st.dataframe(df_hist, use_container_width=True, height=300)

    if "action" in df_hist.columns:
        action_counts = df_hist["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        st.altair_chart(
            alt.Chart(action_counts)
            .mark_bar()
            .encode(
                x="Action:N",
                y="Count:Q",
                color=alt.Color(
                    "Action:N",
                    scale=alt.Scale(
                        domain=[
                            "BUY",
                            "SELL",
                            "CLOSE",
                            "CLOSE_LONG",
                            "CLOSE_SHORT",
                            "HOLD",
                        ],
                        range=[
                            "#22c55e",
                            "#ef4444",
                            "#f59e0b",
                            "#f59e0b",
                            "#f59e0b",
                            "#94a3b8",
                        ],
                    ),
                ),
                tooltip=["Action", "Count"],
            )
            .properties(title="Action distribution", height=200),
            use_container_width=True,
        )


def _render_executed_trades(df_trades: pd.DataFrame) -> None:
    st.subheader("Executed trades (CSV)")
    display_cols = [c for c in df_trades.columns if c != "reasoning"]
    st.dataframe(df_trades[display_cols], use_container_width=True, height=300)

    st.subheader("Reasoning log")
    has_reasoning = "reasoning" in df_trades.columns
    action_filter = st.multiselect(
        "Filter by action",
        options=sorted(df_trades["action"].dropna().unique().tolist()),
        default=[],
        key="reasoning_action_filter",
    )
    df_reason = df_trades.copy()
    if action_filter:
        df_reason = df_reason[df_reason["action"].isin(action_filter)]

    for _, row in (
        df_reason.sort_values("timestamp_iso", ascending=False).head(50).iterrows()
    ):
        ts = str(row.get("timestamp_iso", ""))[:19]
        action = str(row.get("action", "—"))
        symbol_r = str(row.get("symbol", ""))
        source = str(row.get("source", ""))
        pnl = row.get("pnl_usdt", None)
        reasoning = str(row.get("reasoning", "")) if has_reasoning else ""

        icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        pnl_str = (
            f"  P&L: ${float(pnl):+.4f}"
            if pnl and str(pnl).strip() not in ("", "nan")
            else ""
        )
        label = f"{icon} {ts}  ·  {symbol_r}  ·  **{action}**  ·  {source}{pnl_str}"
        st.markdown(label)
        if reasoning and reasoning not in ("nan", ""):
            st.caption(reasoning)
        st.divider()


def render_portfolio(symbol: str) -> None:
    df_closed = load_closed_trades(symbol)
    stats_p = load_statistics(symbol)

    initial_capital = float((stats_p or {}).get("initial_capital", 10_000))
    current_capital = float((stats_p or {}).get("current_capital", initial_capital))
    total_pnl_quote = float((stats_p or {}).get("total_pnl_quote", 0))
    total_pnl_pct = float((stats_p or {}).get("total_pnl_pct", 0))
    win_rate = float((stats_p or {}).get("win_rate", 0))
    total_trades_p = int((stats_p or {}).get("total_trades", 0))
    max_dd = float((stats_p or {}).get("max_drawdown_pct", 0))
    best_trade = float((stats_p or {}).get("best_trade_pct", 0))
    worst_trade = float((stats_p or {}).get("worst_trade_pct", 0))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Initial Capital", f"${initial_capital:,.2f}")
    k2.metric("Current Capital", f"${current_capital:,.2f}", f"{total_pnl_pct:+.2f}%")
    k3.metric("Total P&L", f"${total_pnl_quote:+,.2f}", f"{total_pnl_pct:+.2f}%")
    k4.metric("Win Rate", f"{win_rate:.1f}%", f"{total_trades_p} closed trades")
    k5.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.divider()

    if df_closed.empty:
        st.info(
            "No closed trades found yet. P&L will appear here once positions are closed."
        )
        return

    df_closed["cumulative_pnl"] = df_closed["pnl_quote"].cumsum()
    df_closed["capital"] = initial_capital + df_closed["cumulative_pnl"]
    df_closed["trade_n"] = range(1, len(df_closed) + 1)
    df_closed["win"] = df_closed["pnl_quote"] >= 0

    _render_capital_curve(df_closed)
    _render_pnl_bars(df_closed)

    col_a, col_b, col_c, col_d = st.columns(4)
    wins = df_closed[df_closed["win"]]
    losses = df_closed[~df_closed["win"]]
    col_a.metric(
        "Gross Profit", f"${wins['pnl_quote'].sum():+,.2f}", f"{len(wins)} wins"
    )
    col_b.metric(
        "Gross Loss", f"${losses['pnl_quote'].sum():+,.2f}", f"{len(losses)} losses"
    )
    col_c.metric(
        "Best Trade",
        f"{best_trade:+.2f}%" if stats_p else f"{df_closed['pnl_pct'].max():+.2f}%",
    )
    col_d.metric(
        "Worst Trade",
        f"{worst_trade:+.2f}%" if stats_p else f"{df_closed['pnl_pct'].min():+.2f}%",
    )

    _render_closed_trades_table(df_closed)


def _render_capital_curve(df_closed: pd.DataFrame) -> None:
    st.subheader("Capital curve")
    cap_base = alt.Chart(df_closed).encode(x=alt.X("exit_time:T", title="Time"))
    cap_area = cap_base.mark_area(
        line={"color": "#58a6ff", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#58a6ff", offset=0),
                alt.GradientStop(color="rgba(88,166,255,0)", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
    ).encode(
        y=alt.Y("capital:Q", title="Capital (USDT)"),
        tooltip=[
            alt.Tooltip("exit_time:T", title="Date"),
            alt.Tooltip("capital:Q", title="Capital", format="$,.2f"),
            alt.Tooltip("cumulative_pnl:Q", title="Cum. P&L", format="+,.2f"),
        ],
    )
    cap_points = cap_base.mark_point(size=60, filled=True).encode(
        y=alt.Y("capital:Q"),
        color=alt.Color(
            "win:N",
            scale=alt.Scale(domain=[True, False], range=["#3fb950", "#f85149"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("exit_time:T", title="Date"),
            alt.Tooltip("pnl_quote:Q", title="Trade P&L", format="+,.2f"),
            alt.Tooltip("pnl_pct:Q", title="P&L %", format="+.2f"),
        ],
    )
    st.altair_chart(
        (cap_area + cap_points).properties(height=280).interactive(),
        use_container_width=True,
    )


def _render_pnl_bars(df_closed: pd.DataFrame) -> None:
    st.subheader("P&L per closed trade")
    bar_chart = (
        alt.Chart(df_closed)
        .mark_bar()
        .encode(
            x=alt.X("trade_n:O", title="Trade #"),
            y=alt.Y("pnl_quote:Q", title="P&L (USDT)"),
            color=alt.condition(
                alt.datum.pnl_quote >= 0,
                alt.value("#3fb950"),
                alt.value("#f85149"),
            ),
            tooltip=[
                alt.Tooltip("trade_n:O", title="#"),
                alt.Tooltip("exit_time:T", title="Date"),
                alt.Tooltip("direction:N"),
                alt.Tooltip("entry_price:Q", title="Entry", format=",.4f"),
                alt.Tooltip("exit_price:Q", title="Exit", format=",.4f"),
                alt.Tooltip("pnl_quote:Q", title="P&L (USDT)", format="+,.4f"),
                alt.Tooltip("pnl_pct:Q", title="P&L %", format="+.3f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(bar_chart, use_container_width=True)


def _render_closed_trades_table(df_closed: pd.DataFrame) -> None:
    st.subheader("Closed trades")
    display = df_closed[
        [
            "trade_n",
            "entry_time",
            "exit_time",
            "direction",
            "entry_price",
            "exit_price",
            "quantity",
            "fees",
            "pnl_quote",
            "pnl_pct",
            "confidence",
        ]
    ].copy()
    display.columns = [
        "#",
        "Entry Time",
        "Exit Time",
        "Direction",
        "Entry Price",
        "Exit Price",
        "Qty",
        "Fees",
        "P&L (USDT)",
        "P&L %",
        "Conf",
    ]

    def _color_pnl(val: Any) -> str:
        try:
            return "color: #3fb950" if float(val) >= 0 else "color: #f85149"
        except (TypeError, ValueError):
            return ""

    st.dataframe(
        display.style.map(_color_pnl, subset=["P&L (USDT)", "P&L %"]),
        use_container_width=True,
        height=300,
    )
