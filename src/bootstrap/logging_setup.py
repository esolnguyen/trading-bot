"""Root logging setup and per-engine file-handler routing."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import Settings


_TIMEFRAME_FILE_HANDLER: logging.Handler | None = None


def configure_root_logging() -> None:
    """Set basicConfig level from ``LOGGER_DEBUG`` and silence noisy libraries.

    Must run before any torch / sentence-transformers import so the OpenMP
    duplicate-library segfault on macOS is avoided.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    load_dotenv(override=False)
    debug = os.getenv("LOGGER_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # ChromaDB posthog telemetry errors are harmless version-mismatch noise.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def attach_timeframe_file_handler(settings: Settings) -> None:
    """Route runtime logs to ``<log_dir>/<engine>[/<tf>]/<YYYY-MM-DD>/app.log``."""
    global _TIMEFRAME_FILE_HANDLER

    engine = settings.trading_engine or "default"
    target_dir = Path(settings.log_dir) / engine
    if engine == "scorer" and settings.timeframe:
        target_dir = target_dir / settings.timeframe
    target_dir = target_dir / datetime.now().strftime("%Y-%m-%d")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "app.log"

    root = logging.getLogger()
    if _TIMEFRAME_FILE_HANDLER is not None:
        root.removeHandler(_TIMEFRAME_FILE_HANDLER)
        try:
            _TIMEFRAME_FILE_HANDLER.close()
        except Exception:  # noqa: BLE001
            pass

    handler = logging.FileHandler(target_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [engine=" + engine
            + ((" tf=" + settings.timeframe) if settings.timeframe else "")
            + "] %(name)s: %(message)s"
        )
    )
    handler.setLevel(root.level)
    root.addHandler(handler)
    _TIMEFRAME_FILE_HANDLER = handler
