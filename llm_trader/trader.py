"""Futures trader — thin re-export shim.

Logic lives in src/services/llm_trader/futures_trader.py.
"""
from src.services.llm_trader.futures_trader import FuturesTrader as FuturesTrader

__all__ = ["FuturesTrader"]
