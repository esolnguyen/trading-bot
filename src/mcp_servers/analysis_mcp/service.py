"""Business logic container for the Analysis MCP server.

Owns long-lived resources (Binance feed handle, indicator calculator,
pattern analyzer, chart generator) and a ``MultiTimeframeAnalyzer`` cache
keyed on trading timeframe. Handlers delegate every network call through
this class.
"""

import asyncio
import logging
from time import time

from src.config import BinanceSettings
from src.mcp_servers.analysis_mcp.indicators import TechnicalIndicators
from src.mcp_servers.shared import (
    BinanceFeed,
    IndicatorCalculator,
    MarketSnapshot,
    OHLCVCandle,
)
from src.mcp_servers.shared.services import (
    ChartGenerator,
    MarketAggregator,
    MultiTimeframeAnalyzer,
    PatternAnalyzer,
    TechnicalAnalyzer,
)

logger = logging.getLogger(__name__)


class AnalysisToolsService:
    """Shared state + helpers used by every Analysis MCP tool."""

    def __init__(self, settings: BinanceSettings) -> None:
        self._settings = settings
        self._feed: BinanceFeed | None = None
        self._feed_lock = asyncio.Lock()
        self._calculator = IndicatorCalculator()
        self._patterns = PatternAnalyzer()
        self._charts = ChartGenerator()
        self._analyzer = TechnicalAnalyzer(self._calculator)
        self._aggregator: MarketAggregator | None = None
        self._mtf_cache: dict[str, MultiTimeframeAnalyzer] = {}

    async def get_feed(self) -> BinanceFeed:
        async with self._feed_lock:
            if self._feed is None:
                feed = BinanceFeed(self._settings)
                await feed.start()
                self._feed = feed
            return self._feed

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[OHLCVCandle]:
        feed = await self.get_feed()
        return await feed.get_ohlcv(symbol, timeframe, limit=limit)

    def compute_indicators(self, candles: list[OHLCVCandle]):
        return self._calculator.compute(candles)

    @staticmethod
    def build_technical_indicators(
        candles: list[OHLCVCandle],
    ) -> TechnicalIndicators:
        """Return a TechnicalIndicators handle populated with ``candles``.

        The facade expects ``[timestamp, open, high, low, close, volume]``
        rows. We don't reuse the feed's OHLCVCandle directly because the
        numba pipeline wants a 2D float array.
        """
        rows = [
            [
                float(c.timestamp),
                float(c.open),
                float(c.high),
                float(c.low),
                float(c.close),
                float(c.volume),
            ]
            for c in candles
        ]
        ti = TechnicalIndicators()
        ti.get_data(rows)
        return ti

    def pattern_analyzer(self) -> PatternAnalyzer:
        return self._patterns

    def chart_generator(self) -> ChartGenerator:
        return self._charts

    def technical_analyzer(self) -> TechnicalAnalyzer:
        return self._analyzer

    async def aggregator(self) -> MarketAggregator:
        if self._aggregator is None:
            self._aggregator = MarketAggregator(await self.get_feed())
        return self._aggregator

    async def mtf_analyzer(self, trading_timeframe: str) -> MultiTimeframeAnalyzer:
        if trading_timeframe not in self._mtf_cache:
            feed = await self.get_feed()
            self._mtf_cache[trading_timeframe] = MultiTimeframeAnalyzer(
                feed=feed, trading_timeframe=trading_timeframe
            )
        return self._mtf_cache[trading_timeframe]

    @staticmethod
    def build_snapshot(symbol: str, candles: list[OHLCVCandle]) -> MarketSnapshot:
        """Minimal snapshot so downstream code that expects one is unaffected."""
        price = candles[-1].close if candles else 0.0
        return MarketSnapshot(
            symbol=symbol.upper(),
            price=price,
            change_24h_pct=0.0,
            volume_24h=0.0,
            bid=price,
            ask=price,
            candles=candles,
            timestamp=candles[-1].timestamp if candles else 0,
        )

    @staticmethod
    def freshness_seconds(candles: list[OHLCVCandle]) -> int:
        if not candles:
            return 0
        latest_ms = int(candles[-1].timestamp)
        return max(0, int(time()) - latest_ms // 1000)
