"""Application bootstrap package — dependency wiring and process lifecycle."""

from src.bootstrap.logging_setup import attach_timeframe_file_handler, configure_root_logging
from src.bootstrap.runtime import build_runtime, close_runtime, run_guarded

__all__ = [
    "attach_timeframe_file_handler",
    "build_runtime",
    "close_runtime",
    "configure_root_logging",
    "run_guarded",
]
