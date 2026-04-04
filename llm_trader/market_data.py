"""Fetch the latest N 15-minute OHLCV candles from Binance Futures."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class Candle:
    timestamp: str   # ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataClient:
    """Thin Binance Futures klines client — no auth required for public data."""

    FUTURES_BASE = "https://fapi.binance.com"
    FUTURES_DEMO_BASE = "https://demo-fapi.binance.com"

    def __init__(self, testnet: bool = False, base_url: str = "") -> None:
        self.base_url = base_url or (self.FUTURES_DEMO_BASE if testnet else self.FUTURES_BASE)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "llm-trader/1.0"

    def close(self) -> None:
        self._session.close()

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 1000,
    ) -> list[Candle]:
        """Return up to *limit* closed candles for *symbol* at *interval*."""
        raw: list[list[Any]] = self._get(
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        candles: list[Candle] = []
        for row in raw:
            open_time_ms: int = row[0]
            # Skip the still-open candle (last row whose close_time > now).
            # Binance returns the ongoing candle as the last entry; drop it so
            # the LLM only sees fully-closed bars.
            close_time_ms: int = row[6]
            if close_time_ms > int(time.time() * 1000):
                continue
            candles.append(
                Candle(
                    timestamp=_ms_to_iso(open_time_ms),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms_to_iso(ms: int) -> str:
    """Convert a millisecond UNIX timestamp to an ISO-8601 UTC string."""
    import datetime
    dt = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def candles_to_csv(candles: list[Candle]) -> str:
    """Serialise candles as compact CSV (header + one row per candle)."""
    lines = ["timestamp,open,high,low,close,volume"]
    for c in candles:
        lines.append(f"{c.timestamp},{c.open},{c.high},{c.low},{c.close},{c.volume}")
    return "\n".join(lines)
