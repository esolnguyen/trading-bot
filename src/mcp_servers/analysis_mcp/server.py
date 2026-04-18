"""Analysis MCP server — wiring only.

Responsibilities:

* construct the ``FastMCP`` instance (protocol layer owned by FastMCP),
* build the ``AnalysisToolsService`` that owns the feed + analyzers,
* register tool handlers against the instance.

Transport selection lives in the launcher
(``scripts/run_mcp_analysis.py``) so the same server object can be
mounted under multiple transports without edits here.

Note: avoid ``from __future__ import annotations`` — FastMCP builds
tool schemas via pydantic's TypeAdapter, which can't resolve
stringified hints.
"""

from fastmcp import FastMCP

from src.mcp_servers.analysis_mcp.handlers import register
from src.mcp_servers.analysis_mcp.service import AnalysisToolsService
from src.mcp_servers.base import get_binance_settings


mcp = FastMCP(name="bot-mcp-analysis")
_service = AnalysisToolsService(get_binance_settings())
register(mcp, _service)
