"""Binance feed MCP server — wiring only.

Responsibilities of this module:

* construct the ``FastMCP`` instance (protocol layer owned by FastMCP),
* build the ``BinanceToolsService`` that owns the feed handle,
* register tool handlers against the instance.

Transport selection lives in the launcher
(``scripts/run_mcp_binance.py``) so the same server object can be
mounted under multiple transports without edits here.

Note: avoid ``from __future__ import annotations`` — FastMCP builds
tool schemas via pydantic's TypeAdapter, which can't resolve
stringified hints.
"""

from fastmcp import FastMCP

from src.mcp_servers.base import get_binance_settings
from src.mcp_servers.binance_mcp.handlers import register
from src.mcp_servers.binance_mcp.service import BinanceToolsService


mcp = FastMCP(name="bot-mcp-binance")
_service = BinanceToolsService(get_binance_settings())
register(mcp, _service)
