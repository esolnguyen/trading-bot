"""Lazy-init ``MarketAggregator`` for scheduled ingestion jobs.

``BinanceFeed`` holds a live SDK client and must be ``start()``-ed
inside the running event loop — so we build one per process, cached
behind an ``asyncio.Lock``, and reuse it across job invocations.
Callers never construct either object directly; they call
``get_aggregator(settings)`` from inside an async job.
"""

from __future__ import annotations

import asyncio

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.mcp_servers.shared import BinanceFeed
from src.mcp_servers.shared.services.market_aggregator import MarketAggregator

_feed: BinanceFeed | None = None
_aggregator: MarketAggregator | None = None
_lock = asyncio.Lock()


async def get_aggregator(
    settings: EnrichKnowledgeSettings,
) -> MarketAggregator:
    """Return the process-wide ``MarketAggregator``, starting the feed once."""
    global _feed, _aggregator
    async with _lock:
        if _aggregator is None:
            feed = BinanceFeed(settings.binance)
            await feed.start()
            _feed = feed
            _aggregator = MarketAggregator(feed)
        return _aggregator


async def close() -> None:
    """Tear down the cached feed (used by runner shutdown)."""
    global _feed, _aggregator
    async with _lock:
        if _feed is not None:
            await _feed.close()
        _feed = None
        _aggregator = None
