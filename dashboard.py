"""Streamlit dashboard for the trading bot — live logs + telemetry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.shared.json_io import read_json as _read_json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOG_DIR = Path("logs")
DATA_DIR = Path("data")
OHLCV_DIR = DATA_DIR / "ohlcv"

# Ported from statistics_panel.js STAT_META
STAT_META: dict[str, dict[str, Any]] = {
    "total_trades":               {"label": "Total Trades",       "desc": "Number of closed positions."},
    "winning_trades":             {"label": "Winning Trades",      "desc": "Positions closed with positive P&L."},
    "losing_trades":              {"label": "Losing Trades",       "desc": "Positions closed with negative P&L."},
    "win_rate":                   {"label": "Win Rate",            "desc": "% of trades that were profitable.", "unit": "%", "decimals": 1},
    "total_pnl_pct":              {"label": "Total P&L",           "desc": "Cumulative return across all trades.", "unit": "%", "decimals": 2},
    "avg_trade_pct":              {"label": "Avg Trade P&L",       "desc": "Average per-trade return.", "unit": "%", "decimals": 2},
    "best_trade_pct":             {"label": "Best Trade",          "desc": "Single best trade return.", "unit": "%", "decimals": 2},
    "worst_trade_pct":            {"label": "Worst Trade",         "desc": "Single worst trade return.", "unit": "%", "decimals": 2},
    "sharpe_ratio":               {"label": "Sharpe Ratio",        "desc": "Risk-adjusted return. >1 is good.", "decimals": 2},
    "sortino_ratio":              {"label": "Sortino Ratio",       "desc": "Like Sharpe but penalises downside only. >1 is good.", "decimals": 2},
    "max_drawdown_pct":           {"label": "Max Drawdown",        "desc": "Largest peak-to-trough decline.", "unit": "%", "decimals": 2},
    "profit_factor":              {"label": "Profit Factor",       "desc": "Gross profit / gross loss. >1 is profitable.", "decimals": 2},
    "avg_win_pct":                {"label": "Avg Win",             "desc": "Average P&L of winning trades.", "unit": "%", "decimals": 2},
    "avg_loss_pct":               {"label": "Avg Loss",            "desc": "Average P&L of losing trades.", "unit": "%", "decimals": 2},
    "largest_consecutive_wins":   {"label": "Max Win Streak",      "desc": "Longest consecutive winning run."},
    "largest_consecutive_losses": {"label": "Max Loss Streak",     "desc": "Longest consecutive losing run."},
    "total_pnl_quote":            {"label": "Total P&L ($)",       "desc": "Cumulative P&L in quote currency.", "prefix": "$", "decimals": 2},
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5)
def load_cycle_logs(n: int = 100) -> list[dict]:
    """Return the last *n* cycle log entries from bot.log (JSONL), newest first."""
    path = LOG_DIR / "bot.log"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(entries) >= n:
            break
    return entries


@st.cache_data(ttl=5)
def load_trades() -> pd.DataFrame:
    path = LOG_DIR / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp_iso"])
        return df.sort_values("timestamp_iso", ascending=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_position(symbol: str) -> dict | None:
    slug = symbol.replace("/", "").replace(":", "").lower()
    return _read_json(DATA_DIR / f"position_{slug}.json")


@st.cache_data(ttl=5)
def load_statistics(symbol: str) -> dict | None:
    slug = symbol.replace("/", "").replace(":", "").lower()
    return _read_json(DATA_DIR / f"statistics_{slug}.json")


@st.cache_data(ttl=5)
def load_trade_history(symbol: str) -> pd.DataFrame:
    slug = symbol.replace("/", "").replace(":", "").lower()
    data = _read_json(DATA_DIR / f"trade_history_{slug}.json")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    return df


@st.cache_data(ttl=5)
def load_closed_trades(symbol: str) -> pd.DataFrame:
    """Pair BUY/SELL entries with CLOSE exits and compute per-trade P&L."""
    slug = symbol.replace("/", "").replace(":", "").lower()
    data = _read_json(DATA_DIR / f"trade_history_{slug}.json")
    if not data:
        return pd.DataFrame()

    records = []
    open_entry: dict | None = None
    for r in data:
        action = r.get("action", "")
        if action in ("BUY", "SELL"):
            open_entry = r
        elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_entry:
            entry_price = float(open_entry.get("price") or 0)
            exit_price = float(r.get("price") or 0)
            qty = float(open_entry.get("quantity") or 0)
            entry_fee = float(open_entry.get("fee") or 0)
            exit_fee = float(r.get("fee") or 0)
            direction = "LONG" if open_entry["action"] == "BUY" else "SHORT"
            if entry_price and qty:
                if direction == "LONG":
                    pnl_quote = (exit_price - entry_price) * qty - entry_fee - exit_fee
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_quote = (entry_price - exit_price) * qty - entry_fee - exit_fee
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                records.append({
                    "entry_time": open_entry.get("timestamp"),
                    "exit_time": r.get("timestamp"),
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": qty,
                    "fees": entry_fee + exit_fee,
                    "pnl_quote": round(pnl_quote, 4),
                    "pnl_pct": round(pnl_pct, 3),
                    "confidence": open_entry.get("confidence", ""),
                })
            open_entry = None

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    return df.sort_values("exit_time").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_ohlcv(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    slug = symbol.replace("/", "").replace(":", "").lower()
    path = OHLCV_DIR / f"{slug}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_api_costs() -> dict | None:
    """Load persisted API costs from data/api_costs.json (written by CostStorage)."""
    return _read_json(DATA_DIR / "api_costs.json")


def available_symbols() -> list[str]:
    symbols: set[str] = set()
    for p in DATA_DIR.glob("position_*.json"):
        symbols.add(p.stem.replace("position_", "").upper())
    for p in DATA_DIR.glob("trade_history_*.json"):
        symbols.add(p.stem.replace("trade_history_", "").upper())
    for e in load_cycle_logs(200):
        if "symbol" in e:
            symbols.add(e["symbol"])
    return sorted(symbols) or ["ETHUSDT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_stat(key: str, val: Any) -> str:
    meta = STAT_META.get(key, {})
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val)
    decimals = meta.get("decimals", 0)
    prefix = meta.get("prefix", "")
    unit = meta.get("unit", "")
    return f"{prefix}{num:.{decimals}f}{unit}"


def _stat_color(key: str, val: Any) -> str:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "normal"
    if key == "max_drawdown_pct":
        return "inverse" if num > 0 else "normal"
    if key in ("total_pnl_pct", "total_pnl_quote", "avg_trade_pct", "best_trade_pct", "avg_win_pct"):
        return "normal" if num >= 0 else "inverse"
    if key in ("worst_trade_pct", "avg_loss_pct"):
        return "inverse" if num < 0 else "normal"
    return "normal"


def _position_gauge(position: dict) -> None:
    """Render the SL/TP gauge bar (ported from position_panel.js)."""
    sl = position.get("sl_price") or position.get("stop_loss") or 0
    tp = position.get("tp_price") or position.get("take_profit") or 0
    entry = position.get("entry_price") or 0
    direction = position.get("direction", "BUY")

    if not sl or not tp or not entry:
        return

    # Gauge: 0 = SL, 100 = TP (from position_panel.js calculateGaugePct)
    is_short = direction in ("SELL", "SHORT")
    lo = tp if is_short else sl
    hi = sl if is_short else tp
    current = entry  # best proxy without live price
    if hi != lo:
        pct = max(0.0, min(1.0, (current - lo) / (hi - lo)))
    else:
        pct = 0.5

    st.caption(f"SL {sl:,.4f}  ↔  TP {tp:,.4f}")
    st.progress(pct)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_logs, tab_chart, tab_trades, tab_portfolio, tab_stats, tab_costs = st.tabs(
    ["📋 Cycle Logs", "📈 Price Chart", "💱 Trades", "💼 Portfolio", "📊 Statistics", "💰 API Costs"]
)

# ---- Tab: Cycle Logs -------------------------------------------------------

with tab_logs:
    if not logs:
        st.info("No cycle logs found. Run the bot first.")
    else:
        rows = []
        for e in logs:
            usage = e.get("llm_usage") or {}
            rows.append({
                "Time": e.get("ts", "")[:19],
                "Cycle": e.get("cycle", ""),
                "Symbol": e.get("symbol", ""),
                "Signal": e.get("signal", ""),
                "LLM": e.get("llm_decision", ""),
                "Source": e.get("decision_source", ""),
                "Risk": e.get("risk_outcome", ""),
                "Order": e.get("order_id") or "—",
                "Dry": "✓" if e.get("dry_run") else "",
                "Tokens": usage.get("total_tokens", 0),
                "RAG docs": e.get("rag_docs_retrieved", 0),
                "Error": "⚠" if e.get("llm_error") else "",
            })
        df_logs = pd.DataFrame(rows)

        def _color_decision(val: str) -> str:
            return {"BUY": "color: #22c55e", "SELL": "color: #ef4444", "HOLD": "color: #94a3b8"}.get(val, "")

        st.dataframe(
            df_logs.style.map(_color_decision, subset=["LLM", "Signal"]),
            use_container_width=True,
            height=300,
        )

        if len(rows) > 1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Decision distribution")
                dist = pd.DataFrame(rows)["LLM"].value_counts().reset_index()
                dist.columns = ["Decision", "Count"]
                st.altair_chart(
                    alt.Chart(dist).mark_arc(innerRadius=40).encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color(
                            "Decision:N",
                            scale=alt.Scale(
                                domain=["BUY", "SELL", "HOLD"],
                                range=["#22c55e", "#ef4444", "#94a3b8"],
                            ),
                        ),
                        tooltip=["Decision", "Count"],
                    ).properties(height=220),
                    use_container_width=True,
                )
            with col_b:
                st.subheader("Token usage per cycle")
                df_tok = pd.DataFrame(rows)[["Cycle", "Tokens"]].copy()
                df_tok["Cycle"] = df_tok["Cycle"].astype(str)
                st.altair_chart(
                    alt.Chart(df_tok).mark_line(point=True).encode(
                        x=alt.X("Cycle:O", sort=None, title="Cycle"),
                        y=alt.Y("Tokens:Q"),
                        tooltip=["Cycle", "Tokens"],
                    ).properties(height=220),
                    use_container_width=True,
                )

        st.subheader("Cycle details")
        for e in logs[:20]:
            ts = e.get("ts", "")[:19]
            cycle = e.get("cycle", "?")
            decision = e.get("llm_decision", "?")
            signal = e.get("signal", "?")
            icon = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "⚪"
            with st.expander(f"{icon} Cycle {cycle} · {ts} · Signal={signal} → {decision}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Patterns:**", ", ".join(e.get("patterns") or []) or "none")
                    st.write("**Risk outcome:**", e.get("risk_outcome", "—"))
                    st.write("**Source:**", e.get("decision_source", "—"))
                    if e.get("order_id"):
                        st.write("**Order ID:**", e["order_id"])
                    if e.get("llm_error"):
                        st.error(f"LLM error: {e['llm_error']}")
                with c2:
                    usage = e.get("llm_usage") or {}
                    st.write(
                        "**Tokens:** prompt=%d  completion=%d  total=%d"
                        % (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
                    )
                    st.write("**RAG docs:**", e.get("rag_docs_retrieved", 0))

                reasoning = e.get("decision_reasoning", "")
                if reasoning:
                    st.write("**Reasoning:**")
                    st.info(reasoning)

                # System prompt (extracted from llm_prompt.system_prompt)
                llm_prompt = e.get("llm_prompt") or {}
                sys_prompt = llm_prompt.get("system_prompt", "")
                if sys_prompt:
                    st.write("**System prompt:**")
                    st.text(sys_prompt)

                raw = e.get("llm_raw_response", "")
                if raw:
                    st.write("**Raw LLM response:**")
                    st.code(raw, language="json")


# ---- Tab: Price Chart ------------------------------------------------------

with tab_chart:
    df_ohlcv = load_ohlcv(symbol, ohlcv_tf)
    if df_ohlcv.empty:
        st.info(f"No OHLCV data found for {symbol} / {ohlcv_tf} in {OHLCV_DIR}")
    else:
        df_plot = df_ohlcv.tail(180).copy()
        df_plot["color"] = df_plot.apply(lambda r: "up" if r["close"] >= r["open"] else "down", axis=1)

        base = alt.Chart(df_plot).encode(x=alt.X("timestamp:T", title="Time"))
        wick = base.mark_rule().encode(
            y=alt.Y("low:Q", title="Price (USDT)"),
            y2="high:Q",
            color=alt.Color("color:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
            tooltip=["timestamp:T", "open:Q", "high:Q", "low:Q", "close:Q", "volume:Q"],
        )
        body = base.mark_bar(size=4).encode(
            y="open:Q",
            y2="close:Q",
            color=alt.Color("color:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
        )
        st.altair_chart(
            (wick + body).properties(
                title=f"{symbol} — {ohlcv_tf} candlestick (last {len(df_plot)} candles)", height=400
            ).interactive(),
            use_container_width=True,
        )

        st.altair_chart(
            alt.Chart(df_plot).mark_bar(opacity=0.6).encode(
                x=alt.X("timestamp:T"),
                y=alt.Y("volume:Q", title="Volume"),
                color=alt.Color("color:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
                tooltip=["timestamp:T", "volume:Q"],
            ).properties(height=120).interactive(),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        last_price = df_plot["close"].iloc[-1]
        prev_price = df_plot["close"].iloc[-2] if len(df_plot) > 1 else last_price
        pct = (last_price - prev_price) / prev_price * 100 if prev_price else 0
        col_a.metric("Last close", f"${last_price:,.2f}", f"{pct:+.2f}%")
        col_b.metric("Candles loaded", f"{len(df_ohlcv):,}", f"showing last {len(df_plot)}")


# ---- Tab: Trades -----------------------------------------------------------

with tab_trades:
    df_hist = load_trade_history(symbol)
    df_trades = load_trades()

    # Cumulative P&L chart (ported from performance_chart.js)
    pnl_col = None
    df_for_chart = pd.DataFrame()
    if not df_hist.empty:
        for col in ("pnl_pct", "pnl", "realized_pnl_pct"):
            if col in df_hist.columns:
                pnl_col = col
                break
    if pnl_col and not df_hist.empty:
        ts_col = next((c for c in ("timestamp", "close_time", "time") if c in df_hist.columns), None)
        if ts_col:
            df_for_chart = df_hist[[ts_col, pnl_col]].dropna().sort_values(ts_col).copy()
            df_for_chart["cumulative_pnl"] = pd.to_numeric(df_for_chart[pnl_col], errors="coerce").cumsum()
            df_for_chart["win"] = df_for_chart[pnl_col].apply(lambda v: float(v) >= 0)

    if not df_for_chart.empty:
        st.subheader("Cumulative P&L")
        area_chart = alt.Chart(df_for_chart).mark_area(
            line={"color": "#58a6ff", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#58a6ff", offset=0),
                    alt.GradientStop(color="rgba(88,166,255,0)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        ).encode(
            x=alt.X(f"{ts_col}:T", title="Time"),
            y=alt.Y("cumulative_pnl:Q", title="Cumulative P&L (%)"),
            tooltip=[f"{ts_col}:T", alt.Tooltip("cumulative_pnl:Q", format=".3f"), alt.Tooltip(f"{pnl_col}:Q", format=".3f", title="Trade P&L")],
        ).properties(height=260)

        points = alt.Chart(df_for_chart).mark_point(size=60, filled=True).encode(
            x=alt.X(f"{ts_col}:T"),
            y=alt.Y("cumulative_pnl:Q"),
            color=alt.Color(
                "win:N",
                scale=alt.Scale(domain=[True, False], range=["#3fb950", "#f85149"]),
                legend=None,
            ),
            tooltip=[f"{ts_col}:T", alt.Tooltip("cumulative_pnl:Q", format=".3f")],
        )
        st.altair_chart((area_chart + points).interactive(), use_container_width=True)

    if not df_hist.empty:
        st.subheader("Trade history")
        st.dataframe(df_hist, use_container_width=True, height=300)

        if "action" in df_hist.columns:
            action_counts = df_hist["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            st.altair_chart(
                alt.Chart(action_counts).mark_bar().encode(
                    x="Action:N",
                    y="Count:Q",
                    color=alt.Color(
                        "Action:N",
                        scale=alt.Scale(
                            domain=["BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"],
                            range=["#22c55e", "#ef4444", "#f59e0b", "#f59e0b", "#f59e0b", "#94a3b8"],
                        ),
                    ),
                    tooltip=["Action", "Count"],
                ).properties(title="Action distribution", height=200),
                use_container_width=True,
            )
    elif not df_trades.empty:
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

        for _, row in df_reason.sort_values("timestamp_iso", ascending=False).head(50).iterrows():
            ts = str(row.get("timestamp_iso", ""))[:19]
            action = str(row.get("action", "—"))
            symbol_r = str(row.get("symbol", ""))
            source = str(row.get("source", ""))
            pnl = row.get("pnl_usdt", None)
            reasoning = str(row.get("reasoning", "")) if has_reasoning else ""

            icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            pnl_str = f"  P&L: ${float(pnl):+.4f}" if pnl and str(pnl).strip() not in ("", "nan") else ""
            label = f"{icon} {ts}  ·  {symbol_r}  ·  **{action}**  ·  {source}{pnl_str}"
            st.markdown(label)
            if reasoning and reasoning not in ("nan", ""):
                st.caption(reasoning)
            st.divider()
    else:
        st.info("No trade records found yet.")


# ---- Tab: Portfolio --------------------------------------------------------

with tab_portfolio:
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
        st.info("No closed trades found yet. P&L will appear here once positions are closed.")
    else:
        # Capital curve
        df_closed["cumulative_pnl"] = df_closed["pnl_quote"].cumsum()
        df_closed["capital"] = initial_capital + df_closed["cumulative_pnl"]
        df_closed["trade_n"] = range(1, len(df_closed) + 1)
        df_closed["win"] = df_closed["pnl_quote"] >= 0

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
                x1=1, x2=1, y1=1, y2=0,
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
        st.altair_chart((cap_area + cap_points).properties(height=280).interactive(), use_container_width=True)

        # Per-trade P&L bars
        st.subheader("P&L per closed trade")
        bar_chart = alt.Chart(df_closed).mark_bar().encode(
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
        ).properties(height=220)
        st.altair_chart(bar_chart, use_container_width=True)

        # Win/loss breakdown row
        col_a, col_b, col_c, col_d = st.columns(4)
        wins = df_closed[df_closed["win"]]
        losses = df_closed[~df_closed["win"]]
        col_a.metric("Gross Profit", f"${wins['pnl_quote'].sum():+,.2f}", f"{len(wins)} wins")
        col_b.metric("Gross Loss", f"${losses['pnl_quote'].sum():+,.2f}", f"{len(losses)} losses")
        col_c.metric("Best Trade", f"{best_trade:+.2f}%" if stats_p else f"{df_closed['pnl_pct'].max():+.2f}%")
        col_d.metric("Worst Trade", f"{worst_trade:+.2f}%" if stats_p else f"{df_closed['pnl_pct'].min():+.2f}%")

        # Closed trades table
        st.subheader("Closed trades")
        display = df_closed[[
            "trade_n", "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "quantity", "fees", "pnl_quote", "pnl_pct", "confidence",
        ]].copy()
        display.columns = [
            "#", "Entry Time", "Exit Time", "Direction",
            "Entry Price", "Exit Price", "Qty", "Fees", "P&L (USDT)", "P&L %", "Conf",
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


# ---- Tab: Statistics -------------------------------------------------------

with tab_stats:
    stats = load_statistics(symbol)

    # Position detail with SL/TP gauge (ported from position_panel.js)
    if position:
        st.subheader("Open Position")
        direction = position.get("direction", "—")
        icon = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Side", f"{icon} {direction}")
        pc2.metric("Quantity", position.get("size") or "—")
        entry_price = position.get("entry_price") or 0
        pc3.metric("Entry Price", f"${entry_price:,.4f}" if entry_price else "—")
        sl = position.get("sl_price") or 0
        tp = position.get("tp_price") or 0
        pc4.metric("SL / TP", f"{sl or '—'} / {tp or '—'}")
        _position_gauge(position)
        st.divider()

    if not stats:
        st.info("No statistics file found yet. Close a trade to generate stats.")
    else:
        # Render with proper STAT_META labels (ported from statistics_panel.js)
        known = {k: v for k, v in stats.items() if k in STAT_META and k != "calculation_date"}
        unknown = {k: v for k, v in stats.items() if k not in STAT_META and k != "calculation_date" and isinstance(v, (int, float, str, bool))}

        if known:
            cols = st.columns(4)
            for i, (key, val) in enumerate(known.items()):
                meta = STAT_META[key]
                with cols[i % 4]:
                    formatted = _fmt_stat(key, val)
                    st.metric(meta["label"], formatted, help=meta.get("desc", ""))

        if unknown:
            st.divider()
            st.caption("Other metrics")
            cols = st.columns(4)
            for i, (k, v) in enumerate(unknown.items()):
                with cols[i % 4]:
                    label = k.replace("_", " ").title()
                    st.metric(label, f"{v:.4f}" if isinstance(v, float) else str(v))

        if "calculation_date" in stats:
            st.caption(f"Calculated: {stats['calculation_date']}")

    # Log-derived mini-stats
    if logs:
        st.divider()
        st.subheader("Cycle log summary")
        decisions = [e.get("llm_decision", "") for e in logs]
        total = len(decisions)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycles shown", total)
        c2.metric("BUY", decisions.count("BUY"), f"{decisions.count('BUY')/total*100:.0f}%")
        c3.metric("SELL", decisions.count("SELL"), f"{decisions.count('SELL')/total*100:.0f}%")
        c4.metric("HOLD", decisions.count("HOLD"), f"{decisions.count('HOLD')/total*100:.0f}%")

        tokens = [(e.get("llm_usage") or {}).get("total_tokens", 0) for e in logs if e.get("llm_usage")]
        if tokens:
            st.metric("Avg tokens/cycle", f"{sum(tokens)/len(tokens):,.0f}")

        errors = [e for e in logs if e.get("llm_error")]
        if errors:
            st.warning(f"{len(errors)} cycle(s) had LLM errors in the last {total} logs.")


# ---- Tab: API Costs --------------------------------------------------------

with tab_costs:
    costs = load_api_costs()

    if not costs:
        st.info("No API cost data found (data/api_costs.json). Costs are recorded when the bot runs with openrouter or google providers.")
    else:
        providers = ("openrouter", "google", "lmstudio")
        last_reset = costs.get("last_reset")
        if last_reset:
            st.caption(f"Last reset: {last_reset[:19]}")

        # Summary metrics row
        total_cost = sum(
            (costs.get(p) or {}).get("total_cost", 0.0) for p in providers
        )
        total_in = sum(
            (costs.get(p) or {}).get("total_input_tokens", 0) for p in providers
        )
        total_out = sum(
            (costs.get(p) or {}).get("total_output_tokens", 0) for p in providers
        )

        h1, h2, h3 = st.columns(3)
        h1.metric("Total API cost", f"${total_cost:.4f}")
        h2.metric("Total input tokens", f"{total_in:,}")
        h3.metric("Total output tokens", f"{total_out:,}")

        st.divider()
        st.subheader("Per-provider breakdown")

        for provider in providers:
            pdata = costs.get(provider) or {}
            pcost = pdata.get("total_cost", 0.0)
            pin = pdata.get("total_input_tokens", 0)
            pout = pdata.get("total_output_tokens", 0)
            if not pin and not pout and not pcost:
                continue

            with st.expander(f"{provider.title()}  —  ${pcost:.4f}", expanded=True):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Cost", f"${pcost:.6f}")
                cc2.metric("Input tokens", f"{pin:,}")
                cc3.metric("Output tokens", f"{pout:,}")
                if total_cost > 0:
                    pct = pcost / total_cost * 100
                    st.progress(pct / 100, text=f"{pct:.1f}% of total spend")

        # Cost over token usage (token efficiency)
        if total_in + total_out > 0 and total_cost > 0:
            st.divider()
            cost_per_1k = total_cost / ((total_in + total_out) / 1000)
            st.metric("Cost per 1,000 tokens", f"${cost_per_1k:.6f}")

        # Token usage breakdown chart
        rows_cost = []
        for p in providers:
            pdata = costs.get(p) or {}
            pin = pdata.get("total_input_tokens", 0)
            pout = pdata.get("total_output_tokens", 0)
            if pin or pout:
                rows_cost.append({"Provider": p.title(), "Type": "Input", "Tokens": pin})
                rows_cost.append({"Provider": p.title(), "Type": "Output", "Tokens": pout})

        if rows_cost:
            df_costs = pd.DataFrame(rows_cost)
            st.altair_chart(
                alt.Chart(df_costs).mark_bar().encode(
                    x=alt.X("Provider:N"),
                    y=alt.Y("Tokens:Q"),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Input", "Output"], range=["#58a6ff", "#3fb950"]),
                    ),
                    xOffset="Type:N",
                    tooltip=["Provider", "Type", "Tokens"],
                ).properties(title="Token usage by provider", height=220),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
