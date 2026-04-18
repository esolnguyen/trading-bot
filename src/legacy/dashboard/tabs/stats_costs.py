"""Render the Statistics and API Costs tabs."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.legacy.dashboard.config import STAT_META
from src.legacy.dashboard.formatters import fmt_stat, position_gauge
from src.legacy.dashboard.loaders import load_api_costs, load_statistics


def render_stats(symbol: str, position: dict | None, logs: list[dict]) -> None:
    stats = load_statistics(symbol)

    if position:
        _render_open_position(position)

    if not stats:
        st.info("No statistics file found yet. Close a trade to generate stats.")
    else:
        _render_stat_grid(stats)

    if logs:
        _render_cycle_log_summary(logs)


def _render_open_position(position: dict) -> None:
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
    position_gauge(position)
    st.divider()


def _render_stat_grid(stats: dict) -> None:
    known = {
        k: v for k, v in stats.items() if k in STAT_META and k != "calculation_date"
    }
    unknown = {
        k: v
        for k, v in stats.items()
        if k not in STAT_META
        and k != "calculation_date"
        and isinstance(v, (int, float, str, bool))
    }

    if known:
        cols = st.columns(4)
        for i, (key, val) in enumerate(known.items()):
            meta = STAT_META[key]
            with cols[i % 4]:
                formatted = fmt_stat(key, val)
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


def _render_cycle_log_summary(logs: list[dict]) -> None:
    st.divider()
    st.subheader("Cycle log summary")
    decisions = [e.get("llm_decision", "") for e in logs]
    total = len(decisions)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cycles shown", total)
    c2.metric("BUY", decisions.count("BUY"), f"{decisions.count('BUY')/total*100:.0f}%")
    c3.metric(
        "SELL", decisions.count("SELL"), f"{decisions.count('SELL')/total*100:.0f}%"
    )
    c4.metric(
        "HOLD", decisions.count("HOLD"), f"{decisions.count('HOLD')/total*100:.0f}%"
    )

    tokens = [
        (e.get("llm_usage") or {}).get("total_tokens", 0)
        for e in logs
        if e.get("llm_usage")
    ]
    if tokens:
        st.metric("Avg tokens/cycle", f"{sum(tokens)/len(tokens):,.0f}")

    errors = [e for e in logs if e.get("llm_error")]
    if errors:
        st.warning(f"{len(errors)} cycle(s) had LLM errors in the last {total} logs.")


_COST_PROVIDERS: tuple[str, ...] = ("openrouter", "google", "lmstudio")


def render_api_costs() -> None:
    costs = load_api_costs()

    if not costs:
        st.info(
            "No API cost data found (data/api_costs.json). Costs are recorded "
            "when the bot runs with openrouter or google providers."
        )
        return

    last_reset = costs.get("last_reset")
    if last_reset:
        st.caption(f"Last reset: {last_reset[:19]}")

    total_cost = sum(
        (costs.get(p) or {}).get("total_cost", 0.0) for p in _COST_PROVIDERS
    )
    total_in = sum(
        (costs.get(p) or {}).get("total_input_tokens", 0) for p in _COST_PROVIDERS
    )
    total_out = sum(
        (costs.get(p) or {}).get("total_output_tokens", 0) for p in _COST_PROVIDERS
    )

    h1, h2, h3 = st.columns(3)
    h1.metric("Total API cost", f"${total_cost:.4f}")
    h2.metric("Total input tokens", f"{total_in:,}")
    h3.metric("Total output tokens", f"{total_out:,}")

    st.divider()
    st.subheader("Per-provider breakdown")

    for provider in _COST_PROVIDERS:
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

    if total_in + total_out > 0 and total_cost > 0:
        st.divider()
        cost_per_1k = total_cost / ((total_in + total_out) / 1000)
        st.metric("Cost per 1,000 tokens", f"${cost_per_1k:.6f}")

    rows_cost = []
    for p in _COST_PROVIDERS:
        pdata = costs.get(p) or {}
        pin = pdata.get("total_input_tokens", 0)
        pout = pdata.get("total_output_tokens", 0)
        if pin or pout:
            rows_cost.append({"Provider": p.title(), "Type": "Input", "Tokens": pin})
            rows_cost.append({"Provider": p.title(), "Type": "Output", "Tokens": pout})

    if rows_cost:
        df_costs = pd.DataFrame(rows_cost)
        st.altair_chart(
            alt.Chart(df_costs)
            .mark_bar()
            .encode(
                x=alt.X("Provider:N"),
                y=alt.Y("Tokens:Q"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Input", "Output"], range=["#58a6ff", "#3fb950"]
                    ),
                ),
                xOffset="Type:N",
                tooltip=["Provider", "Type", "Tokens"],
            )
            .properties(title="Token usage by provider", height=220),
            use_container_width=True,
        )
