"""ML inference MCP server — wiring only.

Responsibilities of this module:

* construct the ``FastMCP`` instance (protocol layer owned by FastMCP),
* build the ``MLToolsService`` that owns long-lived resources,
* register tool handlers against the instance.

Transport selection (stdio / HTTP / SSE) lives in the launcher
(``scripts/run_mcp_ml.py``) so the same server object can be mounted under
multiple transports without edits here.

Note: avoid ``from __future__ import annotations`` — FastMCP builds tool
schemas via pydantic's TypeAdapter, which can't resolve stringified hints.
"""

from fastmcp import FastMCP

from src.mcp_servers.base import get_binance_settings, get_storage_settings
from src.mcp_servers.ml_mcp.handlers import register
from src.mcp_servers.ml_mcp.service import MLToolsService


mcp = FastMCP(name="bot-mcp-ml")
_service = MLToolsService(get_binance_settings(), get_storage_settings())
register(mcp, _service)
