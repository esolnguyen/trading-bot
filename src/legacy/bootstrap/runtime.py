"""Runtime graph construction and process-lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.legacy.bootstrap.logging_setup import attach_timeframe_file_handler
from src.legacy.bootstrap.services_factory import (
    try_build_brain_service,
    try_build_discord_notifier,
    try_build_memory_service,
    try_build_statistics_service,
    try_build_trading_strategy,
    try_build_vector_memory,
)
from src.mcp_servers.config import Settings
from src.mcp_servers.shared.infrastructure.ai import LLMManager
from src.mcp_servers.shared.infrastructure.binance import BinanceFeed
from src.mcp_servers.rag_mcp.storage import ChromaStore, Persistence
from src.legacy.interfaces.notifiers import ConsoleNotifier, LoggerNotifier
from src.mcp_servers.ml_mcp.services import (
    AnomalyDetector,
    CycleClassifier,
    DirectionClassifier,
    HistoricalPercentileScorer,
    KeyLevelDetector,
    OutcomePredictor,
    SentimentScorer,
)
from src.mcp_servers.shared.services import (
    ChartGenerator,
    IndicatorCalculator,
    MultiTimeframeAnalyzer,
    PatternAnalyzer,
)
from src.legacy.services.analysis import (
    ContextBuilder,
    MarketAggregator,
    TechnicalAnalyzer,
)
from src.legacy.services.rag import IngestionLoop, MemoryManager, RAGRetriever
from src.legacy.services.rag.ohlcv_writer import OHLCVWriter
from src.legacy.services.trading import Executor, RiskManager, TradingLoop

logger = logging.getLogger(__name__)


def build_runtime(settings: Settings) -> dict[str, Any]:
    """Create the application runtime dependency graph."""
    store = ChromaStore(settings.chroma_path)
    feed = BinanceFeed(settings)
    aggregator = MarketAggregator(feed)
    calculator = IndicatorCalculator()
    tech_analyzer = TechnicalAnalyzer(calculator)
    pattern_analyzer = PatternAnalyzer()
    chart_gen = ChartGenerator()
    retriever = RAGRetriever(store, settings)
    llm = LLMManager(settings)
    executor = Executor(settings)
    memory = MemoryManager(store, settings=settings)
    persistence = Persistence(
        log_dir=settings.log_dir,
        data_dir=settings.data_dir,
        timeframe=settings.timeframe,
        trading_engine=settings.trading_engine,
    )
    attach_timeframe_file_handler(settings)
    logger.info(
        "Logging into %s (timeframe=%s, symbols=%s)",
        persistence.log_dir,
        settings.timeframe,
        ",".join(settings.trading_symbols),
    )
    console_notifier = ConsoleNotifier()
    logger_notifier = LoggerNotifier(logger)

    primary_symbol = (
        settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
    )
    percentile_scorer = HistoricalPercentileScorer(
        csv_path=settings.ohlcv_csv_path(primary_symbol, settings.ml_timeframe),
        timeframe=settings.ml_timeframe,
    )
    per_symbol_scorers: dict[str, HistoricalPercentileScorer] = {}
    for sym in settings.trading_symbols:
        if sym.upper() == primary_symbol.upper():
            continue
        per_symbol_scorers[sym.upper()] = HistoricalPercentileScorer(
            csv_path=settings.ohlcv_csv_path(sym, settings.ml_timeframe),
            timeframe=settings.ml_timeframe,
        )

    key_level_detector = KeyLevelDetector(symbols=settings.trading_symbols)
    cycle_classifier = CycleClassifier(
        timeframe=settings.ml_timeframe, symbols=settings.trading_symbols
    )
    direction_classifier = DirectionClassifier(
        timeframe=settings.ml_timeframe, symbols=settings.trading_symbols
    )
    outcome_predictor = OutcomePredictor()
    anomaly_detector = AnomalyDetector(
        timeframe=settings.ml_timeframe, symbols=settings.trading_symbols
    )
    sentiment_scorer = SentimentScorer()
    multi_tf_analyzer = MultiTimeframeAnalyzer(
        feed, trading_timeframe=settings.timeframe
    )

    def _make_writer(sym: str) -> OHLCVWriter:
        scorer = per_symbol_scorers.get(sym.upper(), percentile_scorer)
        return OHLCVWriter(
            path=settings.ohlcv_csv_path(sym, settings.ml_timeframe),
            on_append=scorer.invalidate_cache,
        )

    ohlcv_writers: dict[str, OHLCVWriter] = {
        sym: _make_writer(sym) for sym in settings.trading_symbols
    }
    ohlcv_writer = ohlcv_writers.get(primary_symbol.upper()) or OHLCVWriter(
        path=settings.ohlcv_csv_path(primary_symbol, settings.ml_timeframe),
        on_append=percentile_scorer.invalidate_cache,
    )

    risk = RiskManager(settings, outcome_predictor=outcome_predictor)

    signal_scorer = None
    if settings.trading_engine == "scorer":
        from src.legacy.services.trading.signal_scorer import (
            SignalScorer,
        )  # noqa: PLC0415

        signal_scorer = SignalScorer(
            settings,
            direction_classifier=direction_classifier,
            key_level_detector=key_level_detector,
            outcome_predictor=outcome_predictor,
        )
        logger.info(
            "Trading engine: scorer — deterministic signal scorer (LLM disabled)"
        )
    else:
        logger.info(
            "Trading engine: %s — skills=%s",
            settings.trading_engine,
            ",".join(settings.trader_skills) or "(none)",
        )

    ingestion = IngestionLoop(store, settings, sentiment_scorer=sentiment_scorer)

    vector_memory = try_build_vector_memory(store)
    brain_service = try_build_brain_service(persistence, vector_memory)
    memory_service = try_build_memory_service(persistence, settings)
    statistics_service = try_build_statistics_service(persistence, settings)
    trading_strategy = try_build_trading_strategy(
        risk, persistence, memory_service, statistics_service, brain_service, settings
    )
    discord_notifier = try_build_discord_notifier(settings)
    builder = ContextBuilder(
        settings,
        brain_service=brain_service,
        trading_strategy=trading_strategy,
        memory_service=memory_service,
        cycle_classifier=cycle_classifier,
        minimal_context=(settings.trading_engine == "llm_skills"),
    )

    trading = TradingLoop(
        aggregator,
        tech_analyzer,
        pattern_analyzer,
        chart_gen,
        retriever,
        builder,
        llm,
        risk,
        executor,
        memory,
        persistence,
        settings,
        console_notifier=console_notifier,
        logger_notifier=logger_notifier,
        trading_strategy=trading_strategy,
        brain_service=brain_service,
        memory_service=memory_service,
        statistics_service=statistics_service,
        discord_notifier=discord_notifier,
        anomaly_detector=anomaly_detector,
        percentile_scorer=percentile_scorer,
        direction_classifier=direction_classifier,
        key_level_detector=key_level_detector,
        cycle_classifier=cycle_classifier,
        multi_tf_analyzer=multi_tf_analyzer,
        ohlcv_writer=ohlcv_writer,
        ohlcv_writers=ohlcv_writers,
        per_symbol_scorers=per_symbol_scorers,
        signal_scorer=signal_scorer,
    )

    return {
        "settings": settings,
        "store": store,
        "feed": feed,
        "percentile_scorer": percentile_scorer,
        "key_level_detector": key_level_detector,
        "cycle_classifier": cycle_classifier,
        "direction_classifier": direction_classifier,
        "outcome_predictor": outcome_predictor,
        "anomaly_detector": anomaly_detector,
        "sentiment_scorer": sentiment_scorer,
        "multi_tf_analyzer": multi_tf_analyzer,
        "ohlcv_writer": ohlcv_writer,
        "ohlcv_writers": ohlcv_writers,
        "per_symbol_scorers": per_symbol_scorers,
        "aggregator": aggregator,
        "calculator": calculator,
        "tech_analyzer": tech_analyzer,
        "pattern_analyzer": pattern_analyzer,
        "chart_gen": chart_gen,
        "retriever": retriever,
        "builder": builder,
        "llm": llm,
        "risk": risk,
        "executor": executor,
        "memory": memory,
        "persistence": persistence,
        "console_notifier": console_notifier,
        "logger_notifier": logger_notifier,
        "ingestion": ingestion,
        "trading": trading,
        "vector_memory": vector_memory,
        "brain_service": brain_service,
        "memory_service": memory_service,
        "statistics_service": statistics_service,
        "trading_strategy": trading_strategy,
        "discord_notifier": discord_notifier,
        "signal_scorer": signal_scorer,
    }


async def run_guarded(name: str, runner: Any, *, logger_: logging.Logger) -> None:
    """Run ``runner()`` and log (without re-raising) any non-cancel exception."""
    try:
        await runner()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger_.exception("%s loop crashed but was isolated: %s", name, exc)


async def close_runtime(runtime: dict[str, Any]) -> None:
    """Shutdown known runtime components in order."""
    close_targets = []
    for name in ("trading", "executor", "feed", "llm", "discord_notifier"):
        target = runtime.get(name)
        if target is not None and hasattr(target, "close"):
            close_targets.append(target)

    for target in close_targets:
        try:
            result = target.close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed during runtime shutdown for %s", target.__class__.__name__
            )
