"""Cross-cutting configuration sub-settings.

Each sub-settings class is a ``pydantic_settings.BaseSettings`` that reads
its own slice of the environment. They are composed by the top-level
folders (``mcp_servers``, ``enrich_knowledge``, ``trading_bot``) into
their own process-specific root settings.

Only put a sub-settings class here when it is consumed by at least two
top-level folders — domain-specific configuration belongs in the
consuming folder's own ``config/`` package.
"""

from .binance import BinanceSettings
from .storage import StorageSettings

__all__ = ["BinanceSettings", "StorageSettings"]
