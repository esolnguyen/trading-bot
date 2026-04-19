"""Daemon entrypoint.

Owns the sleep-and-repeat schedule. Per tick it calls
``loop.run_cycle``; on KeyboardInterrupt it drains the in-flight tick
and exits cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from dotenv import load_dotenv

from .config import TradingBotSettings
from .loop import run_cycle

logger = logging.getLogger("trading_bot.runner")


def _configure_logging() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _run_forever(settings: TradingBotSettings) -> None:
    stop = asyncio.Event()

    def _signal() -> None:
        logger.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal)

    logger.info(
        "starting trading_bot: symbols=%s tf=%s interval=%ds mode=%s model=%s",
        settings.trading_symbols,
        settings.primary_timeframe,
        settings.decision_interval_seconds,
        settings.bot_mode,
        settings.llm_model,
    )

    while not stop.is_set():
        try:
            await run_cycle(settings)
        except Exception:  # noqa: BLE001
            logger.exception("tick failed; continuing")
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=settings.decision_interval_seconds
            )
        except asyncio.TimeoutError:
            pass

    logger.info("trading_bot stopped")


def main() -> None:
    _configure_logging()
    load_dotenv(override=True)
    settings = TradingBotSettings()
    settings.assert_runnable()
    if settings.bot_mode == "off":
        logger.warning("BOT_MODE=off — running ticks but ignoring every decision")
    try:
        asyncio.run(_run_forever(settings))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
