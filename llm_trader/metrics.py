"""Backtest metrics — thin re-export shim.

Logic lives in src/domain/trading/backtest_metrics.py.
"""
from src.domain.trading.backtest_metrics import (
    Trade as Trade,
    calc_metrics as calc_metrics,
    trade_stats as trade_stats,
)

__all__ = ["Trade", "trade_stats", "calc_metrics"]
