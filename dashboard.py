"""Streamlit dashboard for the trading bot — live logs + telemetry.

Data loaders, formatters and tab renderers live under :mod:`src.dashboard`.
This file is the streamlit page script and only wires the pieces together.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.dashboard.loaders import (
    available_symbols,
    load_cycle_logs,
    load_position,
)
from src.dashboard.tabs.logs_chart import render_cycle_logs, render_price_chart
from src.dashboard.tabs.stats_costs import render_api_costs, render_stats
from src.dashboard.tabs.trades_portfolio import render_portfolio, render_trades


st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.sidebar:
    st.title("Bot Dashboard")

    symbols = available_symbols()
    symbol = st.selectbox("Symbol", symbols, index=0)
    ohlcv_tf = st.selectbox("Chart timeframe", ["4h", "1d"], index=0)
    log_lines = st.slider("Cycle log history", 10, 200, 50, step=10)

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_secs = st.slider("Refresh interval (s)", 10, 120, 30, step=5)
    st.caption(f"Last render: {pd.Timestamp.now().strftime('%H:%M:%S')}")


st.title("Trading Bot — Live Dashboard")

position = load_position(symbol)
logs = load_cycle_logs(log_lines)
last_cycle = logs[0] if logs else None

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if position:
        direction = position.get("direction", "—")
        size = position.get("size", 0)
        entry = position.get("entry_price") or "?"
        icon = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        st.metric("Position", f"{icon} {direction}", f"qty={size}  entry={entry}")
    else:
        st.metric("Position", "⚪ FLAT", "no open position")

with col2:
    st.metric("Last signal", last_cycle.get("signal", "—") if last_cycle else "—")

with col3:
    st.metric("LLM decision", last_cycle.get("llm_decision", "—") if last_cycle else "—")

with col4:
    if last_cycle:
        usage = last_cycle.get("llm_usage") or {}
        st.metric("Tokens (last cycle)", f"{usage.get('total_tokens', 0):,}")
    else:
        st.metric("Tokens (last cycle)", "—")

with col5:
    if last_cycle:
        ts = last_cycle.get("ts", "")
        st.metric("Cycle #", last_cycle.get("cycle", "?"), ts[:19] if ts else "")
    else:
        st.metric("Cycle #", "—")

st.divider()


tab_logs, tab_chart, tab_trades, tab_portfolio, tab_stats, tab_costs = st.tabs(
    ["📋 Cycle Logs", "📈 Price Chart", "💱 Trades", "💼 Portfolio", "📊 Statistics", "💰 API Costs"]
)

with tab_logs:
    render_cycle_logs(logs)

with tab_chart:
    render_price_chart(symbol, ohlcv_tf)

with tab_trades:
    render_trades(symbol)

with tab_portfolio:
    render_portfolio(symbol)

with tab_stats:
    render_stats(symbol, position, logs)

with tab_costs:
    render_api_costs()


if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
