"""CryptoCompare news — network layer only.

Returns raw articles in a uniform shape; all filtering, truncation,
and document-id generation lives in the transforms layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import requests

_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class NewsArticle:
    """One raw article from the upstream feed."""

    title: str
    url: str
    body: str
    source: str
    published_at: str
    symbol_tags: str


async def fetch(
    api_key: str, session: Any | None = None
) -> list[NewsArticle]:
    """Pull the latest news batch.

    Returns an empty list on empty upstream payload; HTTP errors and
    rate limits propagate so the job can log + skip the cycle.
    """
    client = session or requests
    response = await asyncio.to_thread(
        client.get,
        _URL,
        headers={"authorization": f"Apikey {api_key}"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json().get("Data") or []
    return [
        NewsArticle(
            title=str(row.get("title", "")),
            url=str(row.get("url", "")),
            body=str(row.get("body", "")),
            source=str(row.get("source_info", {}).get("name", "cryptocompare")),
            published_at=str(row.get("published_on", "")),
            symbol_tags=str(row.get("categories", "")),
        )
        for row in rows
    ]
