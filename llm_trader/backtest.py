"""Backtest harness — thin re-export shim.

Logic lives in src/services/backtest/backtester.py.
"""
from src.services.backtest.backtester import (
    Backtester as Backtester,
    DecideFn as DecideFn,
    _mock_decide as _mock_decide,
    main as main,
)

__all__ = ["Backtester", "DecideFn", "_mock_decide", "main"]
