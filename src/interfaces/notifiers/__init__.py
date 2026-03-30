"""Notification interfaces and adapters."""

from .console_notifier import ConsoleNotifier
from .logger_notifier import LoggerNotifier

__all__ = ["ConsoleNotifier", "LoggerNotifier"]

# Optional Discord notifier (requires discord.py to be installed)
try:
    from .base_notifier import BaseNotifier
    from .discord_notifier import DiscordNotifier
    from .filehandler import DiscordFileHandler
    __all__ += ["BaseNotifier", "DiscordNotifier", "DiscordFileHandler"]
except Exception:  # noqa: BLE001
    pass

