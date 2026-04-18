"""Shared paths and metric metadata for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LOG_DIR = Path("logs")
DATA_DIR = Path("data")
OHLCV_DIR = DATA_DIR / "ohlcv"


# Ported from statistics_panel.js STAT_META.
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
