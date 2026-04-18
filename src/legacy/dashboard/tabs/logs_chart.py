"""Render the Cycle Logs and Price Chart tabs."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.legacy.dashboard.config import OHLCV_DIR
from src.legacy.dashboard.loaders import load_ohlcv


def render_cycle_logs(logs: list[dict]) -> None:
    if not logs:
        st.info("No cycle logs found. Run the bot first.")
        return

    rows = []
    for e in logs:
        usage = e.get("llm_usage") or {}
        rows.append(
            {
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
            }
        )
    df_logs = pd.DataFrame(rows)

    def _color_decision(val: str) -> str:
        return {
            "BUY": "color: #22c55e",
            "SELL": "color: #ef4444",
            "HOLD": "color: #94a3b8",
        }.get(val, "")

    st.dataframe(
        df_logs.style.map(_color_decision, subset=["LLM", "Signal"]),
        use_container_width=True,
        height=300,
    )

    if len(rows) > 1:
        _render_log_summary_charts(rows)

    _render_cycle_details(logs)


def _render_log_summary_charts(rows: list[dict]) -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Decision distribution")
        dist = pd.DataFrame(rows)["LLM"].value_counts().reset_index()
        dist.columns = ["Decision", "Count"]
        st.altair_chart(
            alt.Chart(dist)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Decision:N",
                    scale=alt.Scale(
                        domain=["BUY", "SELL", "HOLD"],
                        range=["#22c55e", "#ef4444", "#94a3b8"],
                    ),
                ),
                tooltip=["Decision", "Count"],
            )
            .properties(height=220),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Token usage per cycle")
        df_tok = pd.DataFrame(rows)[["Cycle", "Tokens"]].copy()
        df_tok["Cycle"] = df_tok["Cycle"].astype(str)
        st.altair_chart(
            alt.Chart(df_tok)
            .mark_line(point=True)
            .encode(
                x=alt.X("Cycle:O", sort=None, title="Cycle"),
                y=alt.Y("Tokens:Q"),
                tooltip=["Cycle", "Tokens"],
            )
            .properties(height=220),
            use_container_width=True,
        )


def _render_cycle_details(logs: list[dict]) -> None:
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
                    % (
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                    )
                )
                st.write("**RAG docs:**", e.get("rag_docs_retrieved", 0))

            reasoning = e.get("decision_reasoning", "")
            if reasoning:
                st.write("**Reasoning:**")
                st.info(reasoning)

            llm_prompt = e.get("llm_prompt") or {}
            sys_prompt = llm_prompt.get("system_prompt", "")
            if sys_prompt:
                st.write("**System prompt:**")
                st.text(sys_prompt)

            raw = e.get("llm_raw_response", "")
            if raw:
                st.write("**Raw LLM response:**")
                st.code(raw, language="json")


def render_price_chart(symbol: str, ohlcv_tf: str) -> None:
    df_ohlcv = load_ohlcv(symbol, ohlcv_tf)
    if df_ohlcv.empty:
        st.info(f"No OHLCV data found for {symbol} / {ohlcv_tf} in {OHLCV_DIR}")
        return

    df_plot = df_ohlcv.tail(180).copy()
    df_plot["color"] = df_plot.apply(
        lambda r: "up" if r["close"] >= r["open"] else "down", axis=1
    )

    base = alt.Chart(df_plot).encode(x=alt.X("timestamp:T", title="Time"))
    wick = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Price (USDT)"),
        y2="high:Q",
        color=alt.Color(
            "color:N",
            scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]),
            legend=None,
        ),
        tooltip=["timestamp:T", "open:Q", "high:Q", "low:Q", "close:Q", "volume:Q"],
    )
    body = base.mark_bar(size=4).encode(
        y="open:Q",
        y2="close:Q",
        color=alt.Color(
            "color:N",
            scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]),
            legend=None,
        ),
    )
    st.altair_chart(
        (wick + body)
        .properties(
            title=f"{symbol} — {ohlcv_tf} candlestick (last {len(df_plot)} candles)",
            height=400,
        )
        .interactive(),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(df_plot)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("timestamp:T"),
            y=alt.Y("volume:Q", title="Volume"),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]),
                legend=None,
            ),
            tooltip=["timestamp:T", "volume:Q"],
        )
        .properties(height=120)
        .interactive(),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    last_price = df_plot["close"].iloc[-1]
    prev_price = df_plot["close"].iloc[-2] if len(df_plot) > 1 else last_price
    pct = (last_price - prev_price) / prev_price * 100 if prev_price else 0
    col_a.metric("Last close", f"${last_price:,.2f}", f"{pct:+.2f}%")
    col_b.metric("Candles loaded", f"{len(df_ohlcv):,}", f"showing last {len(df_plot)}")
