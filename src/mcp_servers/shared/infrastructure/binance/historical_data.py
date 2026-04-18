"""Paginated historical OHLCV fetcher for backtests.

Walks the Binance futures ``/fapi/v1/klines`` cursor-style (via the SDK)
to pull arbitrarily long date ranges without hitting the per-request limit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .candles import Candle, row_to_candle

logger = logging.getLogger(__name__)

# Binance interval -> milliseconds per bar
_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def fetch_history(
    client: Any,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    page_limit: int = 1000,
    max_pages: int = 500,
) -> list[Candle]:
    """Fetch all closed candles for ``symbol`` between ``start_ms`` and ``end_ms``.

    Walks the SDK's ``client.klines()`` cursor-style. Drops any still-open
    bar at the tail (close_time > now). ``client`` is a ``UMFutures`` instance.
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    bar_ms = _INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)

    out: list[Candle] = []
    cursor = start_ms
    for _ in range(max_pages):
        rows: list[list[Any]] = client.klines(
            symbol=symbol,
            interval=interval,
            startTime=cursor,
            endTime=end_ms,
            limit=page_limit,
        )
        if not rows:
            break
        for row in rows:
            if row[6] > now_ms:
                continue  # skip in-progress candle
            out.append(row_to_candle(row))
        last_open = rows[-1][0]
        next_cursor = last_open + bar_ms
        if next_cursor >= end_ms or len(rows) < page_limit:
            break
        cursor = next_cursor

    logger.info("fetch_history: %s %s → %d candles", symbol, interval, len(out))
    return out
