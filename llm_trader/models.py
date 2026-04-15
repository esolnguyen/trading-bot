"""Shared data models — thin re-export shim.

Logic lives in src/domain/trading/models.py.
"""
from src.domain.trading.models import OpenPosition as OpenPosition

__all__ = ["OpenPosition"]
