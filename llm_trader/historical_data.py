"""Historical data fetcher — thin re-export shim.

Logic lives in src/infrastructure/binance/historical_data.py.
"""
from src.infrastructure.binance.historical_data import (
    _INTERVAL_MS as _INTERVAL_MS,
    fetch_history as fetch_history,
)

__all__ = ["fetch_history", "_INTERVAL_MS"]
