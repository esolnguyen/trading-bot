"""Optional-service factories that degrade gracefully when modules are missing."""

from __future__ import annotations

import logging
from typing import Any

from src.mcp_servers.config import Settings
from src.mcp_servers.rag_mcp.storage import ChromaStore, Persistence

logger = logging.getLogger(__name__)


def try_build_vector_memory(store: ChromaStore) -> Any | None:
    try:
        from src.mcp_servers.shared.infrastructure.storage.vector_memory import (
            VectorMemoryService,
        )  # noqa: PLC0415

        return VectorMemoryService(store)
    except Exception:  # noqa: BLE001
        logger.debug("VectorMemoryService unavailable", exc_info=True)
        return None


def try_build_brain_service(
    persistence: Persistence, vector_memory: Any | None
) -> Any | None:
    if vector_memory is None:
        return None
    try:
        from src.legacy.services.trading.brain import (
            TradingBrainService,
        )  # noqa: PLC0415

        return TradingBrainService(
            persistence=persistence, vector_memory=vector_memory, logger=logger
        )
    except Exception:  # noqa: BLE001
        logger.debug("TradingBrainService unavailable", exc_info=True)
        return None


def try_build_memory_service(
    persistence: Persistence, settings: Settings
) -> Any | None:
    try:
        from src.legacy.services.trading.memory_service import (
            TradingMemoryService,
        )  # noqa: PLC0415

        return TradingMemoryService(persistence=persistence, logger=logger)
    except Exception:  # noqa: BLE001
        logger.debug("TradingMemoryService unavailable", exc_info=True)
        return None


def try_build_statistics_service(
    persistence: Persistence, settings: Settings
) -> Any | None:
    try:
        from src.legacy.services.trading.statistics.service import (
            TradingStatisticsService,
        )  # noqa: PLC0415

        return TradingStatisticsService(persistence=persistence, logger=logger)
    except Exception:  # noqa: BLE001
        logger.debug("TradingStatisticsService unavailable", exc_info=True)
        return None


def try_build_trading_strategy(
    risk: Any,
    persistence: Persistence,
    memory_service: Any | None,
    statistics_service: Any | None,
    brain_service: Any | None,
    settings: Settings,
) -> Any | None:
    try:
        from src.legacy.services.trading.trading_strategy import (
            TradingStrategy,
        )  # noqa: PLC0415

        symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
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


def try_build_discord_notifier(settings: Settings) -> Any | None:
    if not getattr(settings, "discord_bot_enabled", False):
        return None
    try:
        import discord  # noqa: PLC0415

        from src.mcp_servers.shared.infrastructure.ai.unified_parser import (
            UnifiedParser,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.discord_notifier import (
            DiscordNotifier,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.filehandler import (
            DiscordFileHandler,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.filehandler_components.cleanup_scheduler import (
            CleanupScheduler,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.filehandler_components.message_deleter import (
            MessageDeleter,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.filehandler_components.message_tracker import (
            MessageTracker,
        )  # noqa: PLC0415
        from src.legacy.interfaces.notifiers.filehandler_components.tracking_persistence import (
            TrackingPersistence,
        )  # noqa: PLC0415
        from src.mcp_servers.shared.utils.format_utils import (
            FormatUtils,
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
