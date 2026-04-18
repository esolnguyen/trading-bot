"""Value formatters and small stateful UI widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.legacy.dashboard.config import STAT_META


def fmt_stat(key: str, val: Any) -> str:
    """Return a formatted numeric value honoring STAT_META prefix/unit/decimals."""
    meta = STAT_META.get(key, {})
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val)
    decimals = meta.get("decimals", 0)
    prefix = meta.get("prefix", "")
    unit = meta.get("unit", "")
    return f"{prefix}{num:.{decimals}f}{unit}"


def stat_color(key: str, val: Any) -> str:
    """Return a streamlit ``delta_color`` string based on the metric sign."""
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "normal"
    if key == "max_drawdown_pct":
        return "inverse" if num > 0 else "normal"
    if key in (
        "total_pnl_pct",
        "total_pnl_quote",
        "avg_trade_pct",
        "best_trade_pct",
        "avg_win_pct",
    ):
        return "normal" if num >= 0 else "inverse"
    if key in ("worst_trade_pct", "avg_loss_pct"):
        return "inverse" if num < 0 else "normal"
    return "normal"


def position_gauge(position: dict) -> None:
    """Render the SL/TP gauge bar (ported from position_panel.js)."""
    sl = position.get("sl_price") or position.get("stop_loss") or 0
    tp = position.get("tp_price") or position.get("take_profit") or 0
    entry = position.get("entry_price") or 0
    direction = position.get("direction", "BUY")

    if not sl or not tp or not entry:
        return

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
