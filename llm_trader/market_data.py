"""Market data client — thin re-export shim.

Logic lives in src/infrastructure/binance/market_data.py.
"""
from src.infrastructure.binance.market_data import (
    Candle as Candle,
    MarketDataClient as MarketDataClient,
    _ms_to_iso as _ms_to_iso,
    candles_to_csv as candles_to_csv,
)

__all__ = ["Candle", "MarketDataClient", "candles_to_csv", "_ms_to_iso"]
