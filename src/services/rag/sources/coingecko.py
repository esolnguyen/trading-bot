"""CoinGecko fear and greed ingestion source."""

from __future__ import annotations

import asyncio
from typing import Any

import requests


class CoinGeckoFearGreedSource:
    """Fetch macro sentiment data from CoinGecko."""

    url = "https://api.coingecko.com/api/v3/global"

    def __init__(self, session: Any | None = None) -> None:
        self.session = session or requests

    async def fetch(self) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(self.session.get, self.url, timeout=15)
        response.raise_for_status()
        payload = response.json().get("data", {})
        return [
            {
                "source": "coingecko",
                "metric": "market_cap_change_percentage_24h_usd",
                "value": payload.get("market_cap_change_percentage_24h_usd", 0),
                "narrative": f"Global crypto market cap changed {payload.get('market_cap_change_percentage_24h_usd', 0)}% in 24h.",
                "timestamp": str(payload.get("updated_at", "")),
            }
        ]
