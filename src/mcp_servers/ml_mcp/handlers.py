"""Tool handlers for the ML MCP server.

Handlers are intentionally thin: validate inputs via the function signature
(FastMCP turns these into the public tool schema), delegate work to
``MLToolsService``, and return a pydantic response model (or ``tool_error``)
as a JSON-serializable dict.

Note: do NOT add ``from __future__ import annotations`` here — FastMCP
builds tool schemas via pydantic's TypeAdapter which cannot resolve
stringified hints.
"""

import logging
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from src.mcp_servers.base import tool_error
from src.mcp_servers.ml_mcp.schemas import (
    AnomalyResponse,
    CycleResponse,
    DirectionResponse,
    KeyLevel,
    KeyLevelsResponse,
    MarketConditions,
    OutcomeResponse,
    PercentileResponse,
    SentimentResponse,
    Side,
    Timeframe,
)
from src.mcp_servers.ml_mcp.service import MLToolsService

logger = logging.getLogger(__name__)

_SymbolField = Annotated[
    str,
    Field(
        min_length=3,
        max_length=20,
        description="Binance perpetual symbol, e.g. BTCUSDT.",
    ),
]


def register(mcp: FastMCP, service: MLToolsService) -> None:
    """Register every ML tool against ``mcp``, bound to ``service``."""

    @mcp.tool()
    async def predict_direction(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "15m",
        candle_limit: Annotated[int, Field(ge=50, le=1000)] = 200,
    ) -> dict:
        """Bullish probability for the next candle from the XGBoost direction model."""
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)

            indicators = service.compute_indicators(candles)
            price = candles[-1].close
            classifier = service.direction_classifier(symbol, timeframe)
            prob = classifier.predict_proba(indicators, price=price, symbol=symbol)
            if prob is None:
                return tool_error(
                    "model_unavailable", symbol=symbol, timeframe=timeframe
                )

            delta = abs(prob - 0.5)
            return DirectionResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                probability=round(prob, 4),
                direction="bullish" if prob >= 0.5 else "bearish",
                conviction=(
                    "high" if delta > 0.2 else "moderate" if delta > 0.1 else "weak"
                ),
                reference_price=price,
                freshness_seconds=service.freshness_seconds(candles),
                model_version=f"xgboost_direction_{timeframe}",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("predict_direction failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def detect_anomaly(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "15m",
        candle_limit: Annotated[int, Field(ge=50, le=1000)] = 200,
    ) -> dict:
        """Isolation Forest anomaly flag for current market microstructure.

        ``is_anomaly=True`` means the trading cycle should skip the LLM call.
        """
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)

            indicators = service.compute_indicators(candles)
            snapshot = service.build_snapshot(symbol, candles)
            detector = service.anomaly_detector(symbol, timeframe)
            is_anom = detector.is_anomaly(snapshot, indicators)

            return AnomalyResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                is_anomaly=bool(is_anom),
                reference_price=candles[-1].close,
                freshness_seconds=service.freshness_seconds(candles),
                model_version=f"isolation_forest_{timeframe}",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("detect_anomaly failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def classify_cycle(
        symbol: _SymbolField = "BTCUSDT",
        daily_candles: Annotated[int, Field(ge=100, le=500)] = 365,
    ) -> dict:
        """Macro regime label (BULL_TRENDING / BEAR_TRENDING / …) from daily candles."""
        try:
            candles = await service.fetch_candles(symbol, "1d", daily_candles)
            if len(candles) < 100:
                return tool_error(
                    "insufficient_history",
                    symbol=symbol,
                    daily_candles=len(candles),
                )

            features = service.build_cycle_features(candles)
            classifier = service.cycle_classifier(symbol)
            result = classifier.predict(features, symbol=symbol)
            if result is None:
                return tool_error("model_unavailable", symbol=symbol)

            regime, confidence = result
            return CycleResponse(
                symbol=symbol.upper(),
                regime=regime,
                confidence=round(confidence, 4),
                daily_candles_used=len(candles),
                model_version="regime_classifier_1d",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("classify_cycle failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_key_levels(
        symbol: _SymbolField = "BTCUSDT",
        max_levels: Annotated[int, Field(ge=1, le=20)] = 6,
    ) -> dict:
        """Nearby historical S/R clusters (DBSCAN, 6-month window)."""
        try:
            # Use latest 15m close as the reference price for proximity ranking.
            candles = await service.fetch_candles(symbol, "15m", 50)
            if not candles:
                return tool_error("no_candles", symbol=symbol)
            price = candles[-1].close

            detector = service.key_level_detector(symbol)
            levels_raw = detector.get_levels(symbol)
            if not levels_raw:
                return tool_error("levels_unavailable", symbol=symbol)

            ranked = sorted(levels_raw, key=lambda l: abs(l["center"] - price))[
                :max_levels
            ]
            levels = [
                KeyLevel(
                    type=str(lvl["type"]),
                    center=float(lvl["center"]),
                    low=float(lvl["low"]),
                    high=float(lvl["high"]),
                    touches=int(lvl["touches"]),
                    distance_pct=round((float(lvl["center"]) - price) / price * 100, 3),
                )
                for lvl in ranked
            ]

            return KeyLevelsResponse(
                symbol=symbol.upper(),
                reference_price=price,
                levels=levels,
                freshness_seconds=service.freshness_seconds(candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_key_levels failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def percentile_rank(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "4h",
        candle_limit: Annotated[int, Field(ge=50, le=1000)] = 200,
    ) -> dict:
        """6-month percentile ranks for RSI/ADX/ATR%/volume/momentum.

        Requires a local OHLCV CSV at
        ``<data_dir>/ohlcv/<symbol>_<timeframe>.csv`` populated by the
        backfill / writer pipeline. Returns ``summary`` as a formatted
        multi-line string safe to inject into an LLM prompt.
        """
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)

            indicators = service.compute_indicators(candles)
            price = candles[-1].close
            scorer = service.percentile_scorer(symbol, timeframe)
            summary = scorer.score(indicators, current_price=price)
            if summary is None:
                return tool_error(
                    "percentile_unavailable",
                    symbol=symbol,
                    timeframe=timeframe,
                    hint="missing or sparse OHLCV CSV",
                )

            return PercentileResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                reference_price=price,
                summary=summary,
                freshness_seconds=service.freshness_seconds(candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("percentile_rank failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def predict_outcome(
        symbol: _SymbolField = "BTCUSDT",
        side: Side = "long",
        timeframe: Timeframe = "15m",
        candle_limit: Annotated[int, Field(ge=50, le=1000)] = 200,
    ) -> dict:
        """Logistic-regression win probability for a proposed entry on ``symbol``.

        Market conditions are derived from the current indicator snapshot
        — the ``side`` field is returned for traceability but the current
        model is side-agnostic (trained on historical win outcomes).
        """
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)

            indicators = service.compute_indicators(candles)
            price = candles[-1].close
            conditions = service.build_market_conditions(indicators, price)

            predictor = service.outcome_predictor()
            block, prob = predictor.should_block(conditions)
            if prob is None:
                return tool_error("model_unavailable", symbol=symbol)

            return OutcomeResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                side=side,
                win_probability=round(prob, 4),
                reference_price=price,
                market_conditions=MarketConditions(**conditions),
                blocked_by_threshold=bool(block),
                model_version="outcome_predictor",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("predict_outcome failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def score_sentiment(
        text: Annotated[
            str,
            Field(
                min_length=1,
                max_length=4096,
                description="Free text (news headline / article snippet) to score.",
            ),
        ],
    ) -> dict:
        """FinBERT sentiment score in [-1, +1] for arbitrary text.

        Returns ``label`` bucketed as bullish/neutral/bearish via ±0.2 bands.
        Requires ``transformers`` + ``torch``; returns ``model_unavailable``
        when those optional deps are missing.
        """
        try:
            scorer = service.sentiment_scorer()
            score = scorer.score(text)
            if score is None:
                return tool_error("model_unavailable")

            if score >= 0.2:
                label = "bullish"
            elif score <= -0.2:
                label = "bearish"
            else:
                label = "neutral"

            return SentimentResponse(
                score=score,
                label=label,
                text_length=len(text),
                model_version="finbert",
                truncated=len(text) > 512,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("score_sentiment failed")
            return tool_error(f"{type(exc).__name__}: {exc}")
