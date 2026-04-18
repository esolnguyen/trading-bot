"""Shared helpers for MCP servers."""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import BinanceSettings, StorageSettings
from src.mcp_servers.config import Settings


logger = logging.getLogger("mcp_servers")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Legacy monolithic settings — retained until the trading-loop split."""
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_binance_settings() -> BinanceSettings:
    return BinanceSettings()


@lru_cache(maxsize=1)
def get_storage_settings() -> StorageSettings:
    return StorageSettings()


def tool_error(message: str, **extra: object) -> dict:
    """Uniform error envelope so clients always see the same shape."""
    payload: dict = {"success": False, "error": message}
    if extra:
        payload.update(extra)
    return payload
