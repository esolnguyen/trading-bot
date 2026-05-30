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
from src.features.outcome import (
    bb_position_label,
    trend_label,
    vol_state_label,
)
from src.features.regime import regime_features_from_candles
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

        Delegates to the shared ``compute_regime_features`` used by the
        trainer, so the served vector matches training exactly — including
        a real ADX (the old inline builder hardcoded ``adx_14 = 0``) and a
        consistently-computed EMA slope.
        """
        return regime_features_from_candles(candles, timeframe="1d")

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

        Bucketing thresholds + source indicators come from the shared
        ``features.outcome`` module — the same ones the trainer used — so the
        gate predicts on the feature space it was fit on. (The old version
        keyed ``volume_state`` off ``obv_slope`` and used ADX/Bollinger
        thresholds that differed from training.)
        """
        atr_pct = (indicators.atr / price * 100) if price > 0 else 0.0
        ema_spread = indicators.ema_20 - indicators.ema_50

        bb_width = indicators.bb_upper - indicators.bb_lower
        bb_pos_val = (price - indicators.bb_lower) / bb_width if bb_width > 0 else 0.5

        return {
            "rsi": indicators.rsi_14,
            "adx": indicators.adx,
            "atr_percentage": atr_pct,
            "trend_direction": trend_label(ema_spread, indicators.adx),
            "volume_state": vol_state_label(indicators.vol_ratio),
            "bb_position": bb_position_label(bb_pos_val),
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
