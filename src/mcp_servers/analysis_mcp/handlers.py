"""Tool handlers for the Analysis MCP server.

Handlers are thin: validate inputs via function signature, delegate to
``AnalysisToolsService``, and return a pydantic response model as a
JSON-serializable dict.

Note: do NOT add ``from __future__ import annotations`` — FastMCP builds
tool schemas via pydantic's TypeAdapter which cannot resolve stringified
hints.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from src.mcp_servers.analysis_mcp.indicator_catalog import run_indicator
from src.mcp_servers.analysis_mcp.schemas import (
    CategoryIndicatorResponse,
    ChartResponse,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorsResponse,
    MultiTFResponse,
    PatternResponse,
    SignalResponse,
    SnapshotResponse,
    Timeframe,
)
from src.mcp_servers.analysis_mcp.service import AnalysisToolsService
from src.mcp_servers.base import tool_error

logger = logging.getLogger(__name__)

_SymbolField = Annotated[
    str,
    Field(
        min_length=3,
        max_length=20,
        description="Binance perpetual symbol, e.g. BTCUSDT.",
    ),
]


def register(mcp: FastMCP, service: AnalysisToolsService) -> None:
    """Register every Analysis tool against ``mcp``, bound to ``service``."""

    @mcp.tool()
    async def compute_indicators(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 200,
    ) -> dict:
        """Compute the canonical indicator set (RSI, MACD, BB, EMA, ADX, ATR, …)."""
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles or len(candles) < 50:
                return tool_error(
                    "insufficient_candles",
                    symbol=symbol,
                    timeframe=timeframe,
                    received=len(candles or []),
                )

            indicators = service.compute_indicators(candles)
            price = candles[-1].close
            return IndicatorsResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                reference_price=price,
                indicators=IndicatorSnapshot(
                    rsi_14=indicators.rsi_14,
                    macd_line=indicators.macd_line,
                    macd_signal=indicators.macd_signal,
                    macd_hist=indicators.macd_hist,
                    bb_upper=indicators.bb_upper,
                    bb_mid=indicators.bb_mid,
                    bb_lower=indicators.bb_lower,
                    ema_20=indicators.ema_20,
                    ema_50=indicators.ema_50,
                    volume_sma_20=indicators.volume_sma_20,
                    atr=indicators.atr,
                    adx=indicators.adx,
                    obv_slope=indicators.obv_slope,
                    choppiness=indicators.choppiness,
                    vol_ratio=indicators.vol_ratio,
                    cci_14=indicators.cci_14,
                ),
                freshness_seconds=service.freshness_seconds(candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("compute_indicators failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def analyze_multi_tf(
        symbol: _SymbolField = "BTCUSDT",
        trading_timeframe: Timeframe = "1h",
        current_signal: Annotated[
            str,
            Field(
                max_length=40,
                description="Short tag describing the current-TF signal, e.g. 'RSI 52, BUY'.",
            ),
        ] = "",
    ) -> dict:
        """Build a multi-timeframe alignment summary for the LLM context."""
        try:
            analyzer = await service.mtf_analyzer(trading_timeframe)
            summary = await analyzer.build_summary(symbol, current_signal)
            if not summary:
                return tool_error(
                    "mtf_unavailable", symbol=symbol, timeframe=trading_timeframe
                )
            return MultiTFResponse(
                symbol=symbol.upper(),
                trading_timeframe=trading_timeframe,
                summary=summary,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze_multi_tf failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_snapshot(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 200,
    ) -> dict:
        """Aggregate ticker + order book + OHLCV + funding + open interest."""
        try:
            aggregator = await service.aggregator()
            snapshot = await aggregator.snapshot(
                symbol.upper(), timeframe=timeframe, limit=candle_limit
            )
            return SnapshotResponse(
                symbol=snapshot.symbol,
                timeframe=timeframe,
                price=snapshot.price,
                change_24h_pct=snapshot.change_24h_pct,
                volume_24h=snapshot.volume_24h,
                bid=snapshot.bid,
                ask=snapshot.ask,
                timestamp=snapshot.timestamp,
                candle_count=len(snapshot.candles),
                funding_rate=snapshot.funding_rate,
                open_interest=snapshot.open_interest,
                freshness_seconds=service.freshness_seconds(snapshot.candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_snapshot failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def analyze_signal(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 200,
        choppiness_threshold: Annotated[float, Field(ge=0.0, le=100.0)] = 61.8,
        rsi_strong_buy: Annotated[float, Field(ge=0.0, le=100.0)] = 30.0,
        rsi_buy: Annotated[float, Field(ge=0.0, le=100.0)] = 40.0,
        rsi_sell: Annotated[float, Field(ge=0.0, le=100.0)] = 60.0,
        rsi_strong_sell: Annotated[float, Field(ge=0.0, le=100.0)] = 70.0,
    ) -> dict:
        """Directional signal (STRONG_BUY…STRONG_SELL) + reasoning + market conditions."""
        try:
            aggregator = await service.aggregator()
            snapshot = await aggregator.snapshot(
                symbol.upper(), timeframe=timeframe, limit=candle_limit
            )
            if len(snapshot.candles) < 50:
                return tool_error(
                    "insufficient_candles",
                    symbol=symbol,
                    timeframe=timeframe,
                    received=len(snapshot.candles),
                )

            analyzer = service.technical_analyzer()
            analysis = analyzer.analyze(
                snapshot.symbol,
                snapshot,
                choppiness_threshold=choppiness_threshold,
                rsi_strong_buy=rsi_strong_buy,
                rsi_buy=rsi_buy,
                rsi_sell=rsi_sell,
                rsi_strong_sell=rsi_strong_sell,
            )
            conditions = analyzer.get_market_conditions(snapshot.symbol, snapshot)

            ind = analysis.indicators
            return SignalResponse(
                symbol=analysis.symbol,
                timeframe=timeframe,
                reference_price=snapshot.price,
                signal=analysis.signal,
                reasoning=analysis.reasoning,
                indicators=IndicatorSnapshot(
                    rsi_14=ind.rsi_14,
                    macd_line=ind.macd_line,
                    macd_signal=ind.macd_signal,
                    macd_hist=ind.macd_hist,
                    bb_upper=ind.bb_upper,
                    bb_mid=ind.bb_mid,
                    bb_lower=ind.bb_lower,
                    ema_20=ind.ema_20,
                    ema_50=ind.ema_50,
                    volume_sma_20=ind.volume_sma_20,
                    atr=ind.atr,
                    adx=ind.adx,
                    obv_slope=ind.obv_slope,
                    choppiness=ind.choppiness,
                    vol_ratio=ind.vol_ratio,
                    cci_14=ind.cci_14,
                ),
                market_conditions=conditions,
                freshness_seconds=service.freshness_seconds(snapshot.candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze_signal failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def detect_patterns(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 200,
    ) -> dict:
        """Detect recent price-action patterns (double bottom/top, S/R, engulfing)."""
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)

            result = service.pattern_analyzer().analyze(
                symbol.upper(), candles, timeframe=timeframe
            )
            return PatternResponse(
                symbol=result.symbol,
                timeframe=timeframe,
                patterns=result.patterns,
                support=result.support,
                resistance=result.resistance,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("detect_patterns failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def render_chart(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        candle_limit: Annotated[int, Field(ge=50, le=500)] = 120,
    ) -> dict:
        """Render a candlestick chart with RSI/MACD panels as base64 PNG."""
        try:
            candles = await service.fetch_candles(symbol, timeframe, candle_limit)
            if not candles or len(candles) < 50:
                return tool_error(
                    "insufficient_candles",
                    symbol=symbol,
                    timeframe=timeframe,
                    received=len(candles or []),
                )

            indicators = service.compute_indicators(candles)
            image = service.chart_generator().render(
                symbol.upper(), candles, indicators
            )
            return ChartResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                image_png_b64=image,
                candle_count=len(candles),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("render_chart failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    # ── category indicator tools ─────────────────────────────────────────

    _MomentumIndicator = Literal[
        "rsi",
        "macd",
        "stochastic",
        "roc",
        "momentum",
        "williams_r",
        "tsi",
        "rmi",
        "ppo",
        "coppock_curve",
        "kst",
        "uo",
    ]
    _TrendIndicator = Literal[
        "adx",
        "supertrend",
        "ichimoku",
        "parabolic_sar",
        "vortex",
        "trix",
        "pfe",
        "td_sequential",
    ]
    _VolatilityIndicator = Literal[
        "atr",
        "bollinger",
        "keltner",
        "donchian",
        "chandelier",
        "choppiness",
        "cci",
        "vhf",
        "ebsw",
    ]
    _VolumeIndicator = Literal[
        "obv",
        "obv_slope",
        "mfi",
        "vwap",
        "twap",
        "cmf",
        "force_index",
        "pvt",
        "ad_line",
        "avg_quote_volume",
    ]
    _StructureIndicator = Literal[
        "support_resistance",
        "fibonacci_retracement",
        "fibonacci_bollinger_bands",
        "pivot_points",
        "fibonacci_pivot_points",
    ]
    _StatisticalIndicator = Literal[
        "hurst",
        "zscore",
        "entropy",
        "kurtosis",
        "skew",
        "stdev",
        "variance",
        "quantile",
        "mad",
        "linreg",
        "fear_and_greed",
    ]

    @mcp.tool()
    async def compute_momentum(
        indicator: _MomentumIndicator = "rsi",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None,
            Field(
                description='Overrides for the indicator\'s default params (e.g. {"length": 21} for rsi).',
            ),
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Momentum indicator — rsi, macd, stochastic, tsi, williams_r, uo, …"""
        return await _run_category(
            service,
            "momentum",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )

    @mcp.tool()
    async def compute_trend(
        indicator: _TrendIndicator = "adx",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None, Field(description="Indicator param overrides.")
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Trend indicator — adx, supertrend, ichimoku, parabolic_sar, trix, …"""
        return await _run_category(
            service,
            "trend",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )

    @mcp.tool()
    async def compute_volatility(
        indicator: _VolatilityIndicator = "atr",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None, Field(description="Indicator param overrides.")
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Volatility indicator — atr, bollinger, keltner, donchian, cci, …"""
        return await _run_category(
            service,
            "volatility",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )

    @mcp.tool()
    async def compute_volume(
        indicator: _VolumeIndicator = "obv",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None, Field(description="Indicator param overrides.")
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Volume indicator — obv, mfi, vwap, cmf, force_index, …"""
        return await _run_category(
            service,
            "volume",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )

    @mcp.tool()
    async def compute_structure(
        indicator: _StructureIndicator = "support_resistance",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None, Field(description="Indicator param overrides.")
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Structure indicator — support/resistance, pivot points, fibonacci levels."""
        return await _run_category(
            service,
            "structure",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )

    @mcp.tool()
    async def compute_statistical(
        indicator: _StatisticalIndicator = "zscore",
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        params: Annotated[
            dict[str, float] | None, Field(description="Indicator param overrides.")
        ] = None,
        last_n: Annotated[int, Field(ge=1, le=500)] = 20,
        candle_limit: Annotated[int, Field(ge=50, le=1500)] = 300,
    ) -> dict:
        """Statistical indicator — hurst, zscore, entropy, skew, fear_and_greed, …"""
        return await _run_category(
            service,
            "statistical",
            indicator,
            symbol,
            timeframe,
            params,
            last_n,
            candle_limit,
        )


async def _run_category(
    service: AnalysisToolsService,
    category: str,
    indicator: str,
    symbol: str,
    timeframe: str,
    params: dict[str, Any] | None,
    last_n: int,
    candle_limit: int,
) -> dict:
    try:
        candles = await service.fetch_candles(symbol, timeframe, candle_limit)
        if not candles or len(candles) < 50:
            return tool_error(
                "insufficient_candles",
                symbol=symbol,
                timeframe=timeframe,
                received=len(candles or []),
            )

        ti = service.build_technical_indicators(candles)
        series, merged = run_indicator(ti, category, indicator, params)

        trimmed = [
            IndicatorSeries(
                name=label,
                latest=(values[-1] if values else None),
                values=values[-last_n:],
            )
            for label, values in series
        ]
        return CategoryIndicatorResponse(
            symbol=symbol.upper(),
            timeframe=timeframe,
            category=category,
            indicator=indicator,
            params={
                k: v for k, v in merged.items() if isinstance(v, (int, float, bool))
            },
            series=trimmed,
            freshness_seconds=service.freshness_seconds(candles),
        ).model_dump()
    except ValueError as exc:
        return tool_error(str(exc), category=category, indicator=indicator)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_%s(%s) failed", category, indicator)
        return tool_error(f"{type(exc).__name__}: {exc}")
