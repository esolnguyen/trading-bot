"""Candle dataclass and helpers for Binance kline data.

Thin conversion layer around ``UMFutures.klines()`` — the SDK returns raw
``list[list[Any]]`` rows; this module turns them into the domain ``Candle``
dataclass used by the LLM trader and backtester.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candle:
    timestamp: str   # ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


def ms_to_iso(ms: int) -> str:
    """Convert a millisecond UNIX timestamp to an ISO-8601 UTC string."""
    dt = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def row_to_candle(row: list[Any]) -> Candle:
    """Convert a raw Binance kline row into a Candle."""
    return Candle(
        timestamp=ms_to_iso(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def fetch_latest_candles(
    client: Any,
    symbol: str,
    interval: str = "15m",
    limit: int = 1000,
) -> list[Candle]:
    """Return up to *limit* closed candles for *symbol* via ``UMFutures.klines``.

    Drops any still-open bar at the tail (close_time > now).
    """
    raw = client.klines(symbol=symbol, interval=interval, limit=limit)
    now_ms = int(time.time() * 1000)
    return [row_to_candle(row) for row in raw if row[6] <= now_ms]


def candles_to_csv(candles: list[Candle]) -> str:
    """Serialise candles as compact CSV (header + one row per candle)."""
    lines = ["timestamp,open,high,low,close,volume"]
    for c in candles:
        lines.append(f"{c.timestamp},{c.open},{c.high},{c.low},{c.close},{c.volume}")
    return "\n".join(lines)
