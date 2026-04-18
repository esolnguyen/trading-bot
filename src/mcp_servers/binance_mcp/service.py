"""Business logic container for the Binance MCP server.

Owns the singleton ``BinanceFeed`` handle with async-safe lazy init so
the feed's HTTP client is constructed inside the running event loop.
Handlers delegate every network call through this class — they never
touch the SDK directly.
"""

import asyncio
import logging
from time import time

from src.config import BinanceSettings
from src.mcp_servers.shared import BinanceFeed

logger = logging.getLogger(__name__)


class BinanceToolsService:
    """Shared state + helpers used by every Binance MCP tool."""

    def __init__(self, settings: BinanceSettings) -> None:
        self._settings = settings
        self._feed: BinanceFeed | None = None
        self._feed_lock = asyncio.Lock()

    async def get_feed(self) -> BinanceFeed:
        async with self._feed_lock:
            if self._feed is None:
                feed = BinanceFeed(self._settings)
                await feed.start()
                self._feed = feed
            return self._feed

    @staticmethod
    def freshness_seconds_ms(timestamp_ms: int | float | None) -> int | None:
        """Return seconds since ``timestamp_ms`` (milliseconds epoch).

        Returns ``None`` when the timestamp is missing or zero so the
        response model's ``freshness_seconds`` stays null.
        """
        if not timestamp_ms:
            return None
        return max(0, int(time()) - int(int(timestamp_ms) // 1000))
