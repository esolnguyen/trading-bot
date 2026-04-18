"""Tool handlers for the RAG MCP server.

Handlers are intentionally thin: ``RAGToolsService`` owns the Chroma
handle and freshness parsing; each handler maps the raw ``(document,
metadata)`` pairs onto the matching pydantic response model.

Note: do NOT add ``from __future__ import annotations`` — FastMCP builds
tool schemas via pydantic's TypeAdapter which cannot resolve stringified
hints.
"""

import logging
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from src.mcp_servers.base import tool_error
from src.mcp_servers.rag_mcp.schemas import (
    IngestionStatusResponse,
    MacroItem,
    MacroResponse,
    MemoryItem,
    MemoryResponse,
    NewsItem,
    NewsResponse,
)
from src.mcp_servers.rag_mcp.service import RAGToolsService

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, service: RAGToolsService) -> None:
    """Register every RAG tool against ``mcp``, bound to ``service``."""

    @mcp.tool()
    async def retrieve_news(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=1024,
                description="Freeform semantic query (symbol, theme, event).",
            ),
        ],
        k: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> dict:
        """Top-k most relevant news docs from the ChromaDB ``news`` collection.

        Each item carries ``freshness_seconds`` when the metadata has a
        parseable timestamp — use it to discount stale stories.
        """
        try:
            rows = service.query_collection("news", query, k)
            items: list[NewsItem] = []
            for document, metadata in rows:
                meta = metadata or {}
                raw_ts = meta.get("published_at") or meta.get("timestamp")
                items.append(
                    NewsItem(
                        title=str(meta.get("title", "")),
                        body=str(meta.get("body", document or "")),
                        source=str(meta.get("source", "unknown")),
                        published_at=service.as_iso(raw_ts),
                        sentiment_score=_as_float(meta.get("sentiment_score")),
                        symbol=meta.get("symbol"),
                        freshness_seconds=service.freshness_seconds(raw_ts),
                    )
                )
            return NewsResponse(
                query=query,
                requested=k,
                returned=len(items),
                items=items,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve_news failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def retrieve_macro(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=1024,
                description="Freeform macro query (Fed, CPI, geopolitical, etc.).",
            ),
        ],
        k: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> dict:
        """Top-k macro docs from the ChromaDB ``macro`` collection."""
        try:
            rows = service.query_collection("macro", query, k)
            items: list[MacroItem] = []
            for document, metadata in rows:
                meta = metadata or {}
                raw_ts = meta.get("published_at") or meta.get("timestamp")
                items.append(
                    MacroItem(
                        source=str(meta.get("source", "unknown")),
                        narrative=str(meta.get("narrative", document or "")),
                        published_at=service.as_iso(raw_ts),
                        freshness_seconds=service.freshness_seconds(raw_ts),
                    )
                )
            return MacroResponse(
                query=query,
                requested=k,
                returned=len(items),
                items=items,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve_macro failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def retrieve_memory(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=1024,
                description="Freeform query describing the current setup.",
            ),
        ],
        symbol: Annotated[
            str | None,
            Field(
                default=None,
                min_length=3,
                max_length=20,
                description="Optional filter — only return trades on this symbol.",
            ),
        ] = None,
        k: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> dict:
        """Top-k past trades (``trade_memory`` collection) similar to the query.

        When ``symbol`` is provided, items whose metadata symbol doesn't
        match are dropped post-query. Empty metadata symbols pass through
        (legacy rows without symbol tagging).
        """
        try:
            rows = service.query_collection("trade_memory", query, k)
            target = symbol.upper() if symbol else None
            items: list[MemoryItem] = []
            for document, metadata in rows:
                meta = metadata or {}
                row_symbol = (meta.get("symbol") or "").upper()
                if target and row_symbol and row_symbol != target:
                    continue
                raw_ts = meta.get("timestamp") or meta.get("published_at")
                items.append(
                    MemoryItem(
                        symbol=row_symbol or "UNKNOWN",
                        action=str(meta.get("action", "UNKNOWN")),
                        timestamp=service.as_iso(raw_ts),
                        reasoning=str(meta.get("reasoning", document or "")),
                        outcome_pnl=_as_float(meta.get("outcome_pnl")),
                        freshness_seconds=service.freshness_seconds(raw_ts),
                    )
                )
            return MemoryResponse(
                query=query,
                symbol_filter=target,
                requested=k,
                returned=len(items),
                items=items,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve_memory failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def ingestion_status() -> dict:
        """Per-collection doc counts + latest ingestion timestamp.

        Useful for monitoring whether the ingestion pipeline is alive —
        low counts or old timestamps mean downstream tools will return
        stale data.
        """
        try:
            counts = {name: service.count(name) for name in RAGToolsService.COLLECTIONS}
            latest = {
                name: service.latest_timestamp(name)
                for name in RAGToolsService.COLLECTIONS
            }
            return IngestionStatusResponse(
                chroma_path=service.chroma_path,
                collections=counts,
                latest_timestamps=latest,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingestion_status failed")
            return tool_error(f"{type(exc).__name__}: {exc}")


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
