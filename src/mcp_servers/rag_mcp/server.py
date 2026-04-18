"""RAG retrieval MCP server — wiring only.

Responsibilities of this module:

* construct the ``FastMCP`` instance (protocol layer owned by FastMCP),
* build the ``RAGToolsService`` that owns the Chroma handle,
* register tool handlers against the instance.

Transport selection lives in the launcher (``scripts/run_mcp_rag.py``).

Note: avoid ``from __future__ import annotations`` — FastMCP builds tool
schemas via pydantic's TypeAdapter, which can't resolve stringified hints.
"""

from fastmcp import FastMCP

from src.mcp_servers.base import get_storage_settings
from src.mcp_servers.rag_mcp.handlers import register
from src.mcp_servers.rag_mcp.service import RAGToolsService


mcp = FastMCP(name="bot-mcp-rag")
_service = RAGToolsService(get_storage_settings())
register(mcp, _service)
