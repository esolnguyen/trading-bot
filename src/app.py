"""Application entrypoint for the trading bot.

Actual wiring lives in :mod:`src.bootstrap`. This module only exposes
``main()`` for async embedding and ``run()`` for the ``python -m src.app``
CLI.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.bootstrap import build_runtime, close_runtime, configure_root_logging, run_guarded
from src.core.config import Settings

logger = logging.getLogger(__name__)


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
            run_guarded("ingestion", ingestion.run, logger_=active_logger),
            name="ingestion-loop",
        ),
        asyncio.create_task(trading.run(), name="trading-loop"),
    ]

    if hasattr(trading, "run_position_monitor"):
        tasks.append(
            asyncio.create_task(
                run_guarded(
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
                run_guarded("discord", discord_notifier.start, logger_=active_logger),
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
        await close_runtime(local_runtime)


def run() -> None:
    """Run the app and surface settings failures clearly."""
    configure_root_logging()
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(f"Settings validation failed: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    run()
