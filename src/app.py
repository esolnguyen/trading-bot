"""Application entrypoint for the trading bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.config import Settings
from src.infrastructure.ai import LLMManager
from src.infrastructure.binance import BinanceFeed
from src.infrastructure.storage import ChromaStore, Persistence
from src.interfaces.notifiers import ConsoleNotifier, LoggerNotifier
from src.services.analysis import (
    ChartGenerator,
    ContextBuilder,
    IndicatorCalculator,
    MarketAggregator,
    PatternAnalyzer,
    TechnicalAnalyzer,
)
from src.services.analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.services.ml.historical_percentile import HistoricalPercentileScorer
from src.services.ml.key_level_detector import KeyLevelDetector
from src.services.ml.cycle_classifier import CycleClassifier
from src.services.ml.direction_classifier import DirectionClassifier
from src.services.ml.outcome_predictor import OutcomePredictor
from src.services.ml.anomaly_detector import AnomalyDetector
from src.services.ml.sentiment_scorer import SentimentScorer
from src.services.rag import IngestionLoop, MemoryManager, RAGRetriever
from src.services.rag.ohlcv_writer import OHLCVWriter
from src.services.trading import Executor, RiskManager, TradingLoop


logger = logging.getLogger(__name__)


def _try_build_vector_memory(store: ChromaStore) -> Any | None:
    """Build VectorMemoryService if the module is available."""
    try:
        from src.infrastructure.storage.vector_memory import (
            VectorMemoryService,
        )  # noqa: PLC0415

        return VectorMemoryService(store)
    except Exception:  # noqa: BLE001
        logger.debug("VectorMemoryService unavailable", exc_info=True)
        return None


def _try_build_brain_service(
    persistence: Persistence, vector_memory: Any | None
) -> Any | None:
    if vector_memory is None:
        return None
    try:
        from src.services.trading.brain_service import (
            TradingBrainService,
        )  # noqa: PLC0415

        return TradingBrainService(
            persistence=persistence, vector_memory=vector_memory, logger=logger
        )
    except Exception:  # noqa: BLE001
        logger.debug("TradingBrainService unavailable", exc_info=True)
        return None


def _try_build_memory_service(
    persistence: Persistence, settings: Settings
) -> Any | None:
    try:
        from src.services.trading.memory_service import (
            TradingMemoryService,
        )  # noqa: PLC0415

        return TradingMemoryService(persistence=persistence, logger=logger)
    except Exception:  # noqa: BLE001
        logger.debug("TradingMemoryService unavailable", exc_info=True)
        return None


def _try_build_statistics_service(
    persistence: Persistence, settings: Settings
) -> Any | None:
    try:
        from src.services.trading.statistics_service import (
            TradingStatisticsService,
        )  # noqa: PLC0415

        return TradingStatisticsService(persistence=persistence, logger=logger)
    except Exception:  # noqa: BLE001
        logger.debug("TradingStatisticsService unavailable", exc_info=True)
        return None


def _try_build_trading_strategy(
    risk: Any,
    persistence: Persistence,
    memory_service: Any | None,
    statistics_service: Any | None,
    brain_service: Any | None,
    settings: Settings,
) -> Any | None:
    try:
        from src.services.trading.trading_strategy import (
            TradingStrategy,
        )  # noqa: PLC0415

        symbol = getattr(settings, "crypto_pair", "BTCUSDT")
        return TradingStrategy(
            logger=logger,
            persistence=persistence,
            risk_manager=risk,
            symbol=symbol,
            settings=settings,
            brain_service=brain_service,
            statistics_service=statistics_service,
            memory_service=memory_service,
        )
    except Exception:  # noqa: BLE001
        logger.debug("TradingStrategy unavailable", exc_info=True)
        return None


def _try_build_discord_notifier(settings: Settings) -> Any | None:
    if not getattr(settings, "discord_bot_enabled", False):
        return None
    try:
        import discord  # noqa: PLC0415
        from src.interfaces.notifiers.discord_notifier import (
            DiscordNotifier,
        )  # noqa: PLC0415
        from src.interfaces.notifiers.filehandler import (
            DiscordFileHandler,
        )  # noqa: PLC0415
        from src.infrastructure.ai.unified_parser import UnifiedParser  # noqa: PLC0415
        from src.shared.format_utils import FormatUtils  # noqa: PLC0415

        from src.interfaces.notifiers.filehandler_components.tracking_persistence import (
            TrackingPersistence,
        )  # noqa: PLC0415
        from src.interfaces.notifiers.filehandler_components.message_tracker import (
            MessageTracker,
        )  # noqa: PLC0415
        from src.interfaces.notifiers.filehandler_components.cleanup_scheduler import (
            CleanupScheduler,
        )  # noqa: PLC0415
        from src.interfaces.notifiers.filehandler_components.message_deleter import (
            MessageDeleter,
        )  # noqa: PLC0415

        intents = discord.Intents.default()
        bot = discord.Client(intents=intents)
        persistence = TrackingPersistence(
            tracking_file="data/discord_tracking.json",
            logger=logger,
        )
        tracker = MessageTracker(
            persistence_handler=persistence, logger=logger, config=settings
        )
        scheduler = CleanupScheduler(
            cleanup_interval=getattr(settings, "discord_cleanup_interval", 300),
            logger=logger,
        )
        deleter = MessageDeleter(bot=bot, logger=logger)
        file_handler = DiscordFileHandler(
            bot=bot,
            logger=logger,
            config=settings,
            persistence=persistence,
            tracker=tracker,
            scheduler=scheduler,
            deleter=deleter,
        )
        return DiscordNotifier(
            logger=logger,
            config=settings,
            unified_parser=UnifiedParser(logger=logger),
            formatter=FormatUtils(),
            bot=bot,
            file_handler=file_handler,
        )
    except Exception:  # noqa: BLE001
        logger.debug("DiscordNotifier unavailable", exc_info=True)
        return None


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
    persistence = Persistence(data_dir="data")
    console_notifier = ConsoleNotifier()
    logger_notifier = LoggerNotifier(logger)

    # ML services — all optional, degrade gracefully if model files missing
    percentile_scorer = HistoricalPercentileScorer(
        csv_path=settings.ohlcv_csv_path(settings.crypto_pair, settings.ml_timeframe),
        timeframe=settings.ml_timeframe,
    )
    # Build per-symbol percentile scorers so every traded symbol gets
    # its own rolling 6-month history context.
    per_symbol_scorers: dict[str, HistoricalPercentileScorer] = {}
    for _sym in settings.trading_symbols:
        _path = settings.ohlcv_csv_path(_sym, settings.ml_timeframe)
        if _path != settings.ohlcv_csv_path(settings.crypto_pair, settings.ml_timeframe):
            per_symbol_scorers[_sym.upper()] = HistoricalPercentileScorer(
                csv_path=_path, timeframe=settings.ml_timeframe
            )

    key_level_detector = KeyLevelDetector(symbols=settings.trading_symbols)
    cycle_classifier = CycleClassifier(timeframe=settings.ml_timeframe, symbols=settings.trading_symbols)
    direction_classifier = DirectionClassifier(timeframe=settings.ml_timeframe, symbols=settings.trading_symbols)
    outcome_predictor = OutcomePredictor()
    anomaly_detector = AnomalyDetector(timeframe=settings.ml_timeframe, symbols=settings.trading_symbols)
    sentiment_scorer = SentimentScorer()
    multi_tf_analyzer = MultiTimeframeAnalyzer(feed, trading_timeframe=settings.timeframe)

    # One OHLCVWriter per trading symbol, each wired to invalidate its
    # corresponding HistoricalPercentileScorer cache on every new row.
    def _make_writer(sym: str) -> OHLCVWriter:
        scorer = per_symbol_scorers.get(sym.upper(), percentile_scorer)
        return OHLCVWriter(
            path=settings.ohlcv_csv_path(sym, settings.ml_timeframe),
            on_append=scorer.invalidate_cache,
        )

    ohlcv_writers: dict[str, OHLCVWriter] = {
        sym: _make_writer(sym) for sym in settings.trading_symbols
    }
    # Backwards-compat: expose the primary symbol writer as `ohlcv_writer`
    primary_sym = settings.trading_symbols[0] if settings.trading_symbols else settings.crypto_pair
    ohlcv_writer = ohlcv_writers.get(primary_sym.upper()) or OHLCVWriter(
        path=settings.ohlcv_csv_path(settings.crypto_pair, settings.ml_timeframe),
        on_append=percentile_scorer.invalidate_cache,
    )

    risk = RiskManager(settings, outcome_predictor=outcome_predictor)

    # Deterministic signal scorer (replaces LLM when USE_SIGNAL_SCORER=true)
    signal_scorer = None
    if settings.use_signal_scorer:
        from src.services.trading.signal_scorer import SignalScorer  # noqa: PLC0415
        signal_scorer = SignalScorer(
            settings,
            direction_classifier=direction_classifier,
            key_level_detector=key_level_detector,
            outcome_predictor=outcome_predictor,
        )
        logger.info("Signal scorer enabled — LLM will not be used for decisions")

    ingestion = IngestionLoop(store, settings, sentiment_scorer=sentiment_scorer)

    # Advanced services (conditional on availability)
    vector_memory = _try_build_vector_memory(store)
    brain_service = _try_build_brain_service(persistence, vector_memory)
    memory_service = _try_build_memory_service(persistence, settings)
    statistics_service = _try_build_statistics_service(persistence, settings)
    trading_strategy = _try_build_trading_strategy(
        risk, persistence, memory_service, statistics_service, brain_service, settings
    )
    discord_notifier = _try_build_discord_notifier(settings)
    builder = ContextBuilder(
        settings,
        brain_service=brain_service,
        trading_strategy=trading_strategy,
        memory_service=memory_service,
        cycle_classifier=cycle_classifier,
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

    rt: dict[str, Any] = {
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

    return rt


async def _run_guarded(name: str, runner: Any, *, logger_: logging.Logger) -> None:
    try:
        await runner()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger_.exception("%s loop crashed but was isolated: %s", name, exc)


async def _close_runtime(runtime: dict[str, Any]) -> None:
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


async def main(
    *,
    runtime: dict[str, Any] | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    """Wire and run the ingestion and trading loops concurrently."""
    active_logger = logger_ or logger
    local_runtime = runtime

    if local_runtime is None:
        settings = Settings.from_env()
        local_runtime = build_runtime(settings)

    trading = local_runtime["trading"]
    ingestion = local_runtime["ingestion"]
    discord_notifier = local_runtime.get("discord_notifier")

    # Reconcile saved position against live exchange state before starting
    _trading_strategy = local_runtime.get("trading_strategy")
    _ts = getattr(_trading_strategy, "trading_strategy", None) or _trading_strategy
    _feed = local_runtime.get("feed")
    if _ts is not None and _feed is not None and hasattr(_ts, "reconcile"):
        try:
            await _ts.reconcile(_feed)
        except Exception:  # noqa: BLE001
            active_logger.warning(
                "Startup position reconciliation failed", exc_info=True
            )

    tasks = [
        asyncio.create_task(
            _run_guarded("ingestion", ingestion.run, logger_=active_logger),
            name="ingestion-loop",
        ),
        asyncio.create_task(trading.run(), name="trading-loop"),
    ]

    if hasattr(trading, "run_position_monitor"):
        tasks.append(
            asyncio.create_task(
                _run_guarded(
                    "position-monitor",
                    trading.run_position_monitor,
                    logger_=active_logger,
                ),
                name="position-monitor",
            )
        )

    if discord_notifier is not None and hasattr(discord_notifier, "start"):
        tasks.append(
            asyncio.create_task(
                _run_guarded("discord", discord_notifier.start, logger_=active_logger),
                name="discord-notifier",
            )
        )

    trading_task = tasks[1]

    try:
        await trading_task
    except asyncio.CancelledError:
        raise
    finally:
        if hasattr(trading, "stop"):
            trading.stop()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _close_runtime(local_runtime)


def run() -> None:
    """Run the app and surface settings failures clearly."""
    # Must be set before any torch / sentence-transformers import to prevent the
    # OpenMP duplicate-library segfault on macOS (Apple Silicon + ChromaDB).
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Determine log level before loading full settings (avoid double-load)
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
    _debug = _os.getenv("LOGGER_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    _level = logging.DEBUG if _debug else logging.INFO
    logging.basicConfig(
        level=_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # ChromaDB posthog telemetry errors are harmless version-mismatch noise — suppress them
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(f"Settings validation failed: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    run()
