"""Structured logger-backed notifier."""

from __future__ import annotations

import logging


class LoggerNotifier:
    """Send human-readable updates to a Python logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def notify(self, message: str) -> None:
        self.logger.info(message)
