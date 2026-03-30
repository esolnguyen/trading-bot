"""Alternative.me ingestion source."""

from __future__ import annotations

import asyncio
from typing import Any

import requests


class AlternativeMeSource:
    """Fetch the fear and greed index from Alternative.me."""

    url = "https://api.alternative.me/fng/"

    def __init__(self, session: Any | None = None) -> None:
        self.session = session or requests

    async def fetch(self) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(self.session.get, self.url, timeout=15)
        response.raise_for_status()
        payload = response.json().get("data", [{}])[0]
        return [
            {
                "source": "alternative_me",
                "metric": "fear_and_greed",
                "value": payload.get("value", ""),
                "narrative": f"Fear and Greed Index is {payload.get('value', '')} ({payload.get('value_classification', 'unknown')}).",
                "timestamp": str(payload.get("timestamp", "")),
            }
        ]
