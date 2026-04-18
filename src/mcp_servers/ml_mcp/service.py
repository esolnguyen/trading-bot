"""Business logic container for the ML MCP server.

Owns long-lived resources (Binance feed, indicator calculator, per-symbol
model caches) so tool handlers stay thin. Every tool added to
``handlers.py`` should delegate through this class rather than reaching
into infrastructure directly.
"""

import asyncio
import logging
from time import time
from typing import Any

from src.config import BinanceSettings, StorageSettings
from src.mcp_servers.shared import (
    BinanceFeed,
    IndicatorCalculator,
    MarketSnapshot,
    OHLCVCandle,
)
from src.mcp_servers.ml_mcp.services import (
    AnomalyDetector,
    CycleClassifier,
    DirectionClassifier,
    HistoricalPercentileScorer,
    KeyLevelDetector,
    OutcomePredictor,
    SentimentScorer,
)

logger = logging.getLogger(__name__)


class MLToolsService:
    """Shared state + services used by every ML MCP tool."""

    def __init__(self, binance: BinanceSettings, storage: StorageSettings) -> None:
        self._binance = binance
        self._storage = storage
        self._feed: BinanceFeed | None = None
        self._feed_lock = asyncio.Lock()
        self._calculator = IndicatorCalculator()

        # Per-(symbol, timeframe) model caches.
        self._direction_cache: dict[str, DirectionClassifier] = {}
        self._anomaly_cache: dict[str, AnomalyDetector] = {}
        self._cycle_cache: dict[str, CycleClassifier] = {}
        self._percentile_cache: dict[str, HistoricalPercentileScorer] = {}
        self._key_level_cache: dict[str, KeyLevelDetector] = {}

        # Singletons — no per-symbol specialisation.
        self._outcome: OutcomePredictor | None = None
        self._sentiment: SentimentScorer | None = None

    # ── feed & indicators ────────────────────────────────────────────────

    async def get_feed(self) -> BinanceFeed:
        async with self._feed_lock:
            if self._feed is None:
                feed = BinanceFeed(self._binance)
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

    # ── direction classifier (predict_direction) ─────────────────────────

    def direction_classifier(self, symbol: str, timeframe: str) -> DirectionClassifier:
        key = f"{timeframe}|{symbol.upper()}"
        if key not in self._direction_cache:
            self._direction_cache[key] = DirectionClassifier(
                timeframe=timeframe, symbols=[symbol]
            )
        return self._direction_cache[key]

    # ── anomaly detector (detect_anomaly) ────────────────────────────────

    def anomaly_detector(self, symbol: str, timeframe: str) -> AnomalyDetector:
        key = f"{timeframe}|{symbol.upper()}"
        if key not in self._anomaly_cache:
            self._anomaly_cache[key] = AnomalyDetector(
                timeframe=timeframe, symbols=[symbol]
            )
        return self._anomaly_cache[key]

    @staticmethod
    def build_snapshot(symbol: str, candles: list[OHLCVCandle]) -> MarketSnapshot:
        """Minimal snapshot the AnomalyDetector consumes."""
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

    # ── cycle classifier (classify_cycle) ────────────────────────────────

    def cycle_classifier(self, symbol: str) -> CycleClassifier:
        # Regime models are trained on 1d candles.
        key = f"1d|{symbol.upper()}"
        if key not in self._cycle_cache:
            self._cycle_cache[key] = CycleClassifier(timeframe="1d", symbols=[symbol])
        return self._cycle_cache[key]

    @staticmethod
    def build_cycle_features(candles: list[OHLCVCandle]) -> dict[str, float]:
        """Build the daily feature dict that CycleClassifier expects.

        Mirrors ``TradingContext._refresh_regime`` — kept inline to avoid
        coupling MCP to trading loop internals.
        """
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        def ema(series: list[float], p: int) -> float:
            k = 2.0 / (p + 1)
            v = series[0]
            for x in series[1:]:
                v = x * k + v * (1 - k)
            return v

        c = closes[-1]
        e50 = ema(closes[-50:], 50)
        e100 = ema(closes[-100:], 100)
        e200 = ema(closes, 200)
        h52w = max(highs[-365:]) if len(highs) >= 365 else max(highs)
        l52w = min(lows[-365:]) if len(lows) >= 365 else min(lows)

        realized_vol = (
            sum((closes[i] / closes[i - 1] - 1) ** 2 for i in range(-30, 0)) / 30
        ) ** 0.5 * (365**0.5)

        return {
            "ema50_dist": (c - e50) / e50 if e50 else 0.0,
            "ema100_dist": (c - e100) / e100 if e100 else 0.0,
            "ema200_dist": (c - e200) / e200 if e200 else 0.0,
            "ema50_slope": (
                (ema(closes[-10:], 10) - ema(closes[-15:-5], 10)) / e50 if e50 else 0.0
            ),
            "high_52w_dist": (c - h52w) / h52w if h52w else 0.0,
            "low_52w_dist": (c - l52w) / l52w if l52w else 0.0,
            "adx_14": 0.0,
            "realized_vol": realized_vol,
            "hh_count": sum(
                1 for i in range(-89, 0) if i + 1 < 0 and highs[i] > highs[i - 1]
            ),
            "ll_count": sum(
                1 for i in range(-89, 0) if i + 1 < 0 and lows[i] < lows[i - 1]
            ),
        }

    # ── key level detector (get_key_levels) ──────────────────────────────

    def key_level_detector(self, symbol: str) -> KeyLevelDetector:
        key = symbol.upper()
        if key not in self._key_level_cache:
            self._key_level_cache[key] = KeyLevelDetector(symbols=[symbol])
        return self._key_level_cache[key]

    # ── percentile scorer (percentile_rank) ──────────────────────────────

    def percentile_scorer(
        self, symbol: str, timeframe: str
    ) -> HistoricalPercentileScorer:
        key = f"{timeframe}|{symbol.upper()}"
        if key not in self._percentile_cache:
            csv_path = self._storage.ohlcv_csv_path(symbol, timeframe)
            self._percentile_cache[key] = HistoricalPercentileScorer(
                csv_path=csv_path, timeframe=timeframe
            )
        return self._percentile_cache[key]

    # ── outcome predictor (predict_outcome) ──────────────────────────────

    def outcome_predictor(self) -> OutcomePredictor:
        if self._outcome is None:
            self._outcome = OutcomePredictor()
        return self._outcome

    @staticmethod
    def build_market_conditions(indicators: Any, price: float) -> dict[str, Any]:
        """Derive the OutcomePredictor input dict from an indicator snapshot.

        Mirrors ``SignalScorer._build_market_conditions`` so the gate-side
        feature assembly stays identical between trading loop and MCP.
        """
        atr_pct = (indicators.atr / price * 100) if price > 0 else 0.0
        ema_bull = indicators.ema_20 > indicators.ema_50

        if indicators.adx >= 25:
            trend = "BULLISH" if ema_bull else "BEARISH"
        else:
            trend = "NEUTRAL"

        vol_state = "HIGH" if indicators.obv_slope > 0.5 else "NORMAL"

        bb_width = indicators.bb_upper - indicators.bb_lower
        if bb_width > 0:
            bb_pos_val = (price - indicators.bb_lower) / bb_width
            if bb_pos_val > 0.8:
                bb_position = "UPPER"
            elif bb_pos_val < 0.2:
                bb_position = "LOWER"
            else:
                bb_position = "MIDDLE"
        else:
            bb_position = "MIDDLE"

        return {
            "rsi": indicators.rsi_14,
            "adx": indicators.adx,
            "atr_percentage": atr_pct,
            "trend_direction": trend,
            "volume_state": vol_state,
            "bb_position": bb_position,
        }

    # ── sentiment scorer (score_sentiment) ───────────────────────────────

    def sentiment_scorer(self) -> SentimentScorer:
        if self._sentiment is None:
            self._sentiment = SentimentScorer()
        return self._sentiment

    # ── misc ─────────────────────────────────────────────────────────────

    @staticmethod
    def freshness_seconds(candles: list[OHLCVCandle]) -> int:
        if not candles:
            return 0
        latest_ms = int(candles[-1].timestamp)
        return max(0, int(time()) - latest_ms // 1000)
