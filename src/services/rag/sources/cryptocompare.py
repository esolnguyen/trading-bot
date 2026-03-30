"""CryptoCompare news ingestion source."""

from __future__ import annotations

import asyncio
from typing import Any

import requests


class CryptoCompareNewsSource:
    """Fetch crypto news articles from CryptoCompare."""

    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

    def __init__(self, api_key: str, session: Any | None = None) -> None:
        self.api_key = api_key
        self.session = session or requests

    async def fetch(self) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(
            self.session.get,
            self.url,
            headers={"authorization": f"Apikey {self.api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("Data", [])
        return [
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "body": article.get("body", ""),
                "source": article.get("source_info", {}).get("name", "cryptocompare"),
                "published_at": str(article.get("published_on", "")),
                "symbol_tags": article.get("categories", ""),
            }
            for article in articles
        ]
