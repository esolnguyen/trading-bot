"""DefiLlama ingestion source."""

from __future__ import annotations

import asyncio
from typing import Any

import requests


class DefiLlamaSource:
    """Fetch DeFi TVL summary data."""

    url = "https://api.llama.fi/chains"

    def __init__(self, session: Any | None = None) -> None:
        self.session = session or requests

    async def fetch(self) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(self.session.get, self.url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        total_tvl = sum(float(item.get("tvl", 0)) for item in payload)
        return [
            {
                "source": "defillama",
                "metric": "total_tvl",
                "value": total_tvl,
                "narrative": f"Total tracked DeFi TVL is {total_tvl:.2f} USD across reported chains.",
                "timestamp": "",
            }
        ]
