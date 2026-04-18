"""CoinGecko global market — network layer only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import requests

_URL = "https://api.coingecko.com/api/v3/global"
_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class GlobalMarketReading:
    """One CoinGecko ``/global`` sample."""

    market_cap_change_24h_pct: float
    timestamp: str


async def fetch(session: Any | None = None) -> GlobalMarketReading | None:
    """Fetch the latest global-market snapshot from CoinGecko."""
    client = session or requests
    response = await asyncio.to_thread(client.get, _URL, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json().get("data") or {}
    if not payload:
        return None
    return GlobalMarketReading(
        market_cap_change_24h_pct=float(
            payload.get("market_cap_change_percentage_24h_usd", 0) or 0
        ),
        timestamp=str(payload.get("updated_at", "")),
    )
