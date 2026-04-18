"""Pydantic response models for the RAG MCP server.

Every returned doc carries ``freshness_seconds`` when its metadata has a
parseable timestamp, so the LLM can self-discount stale ingestion output.
"""

from pydantic import BaseModel


class NewsItem(BaseModel):
    title: str
    body: str
    source: str
    published_at: str | None = None
    sentiment_score: float | None = None
    symbol: str | None = None
    freshness_seconds: int | None = None


class NewsResponse(BaseModel):
    success: bool = True
    query: str
    requested: int
    returned: int
    items: list[NewsItem]


class MacroItem(BaseModel):
    source: str
    narrative: str
    published_at: str | None = None
    freshness_seconds: int | None = None


class MacroResponse(BaseModel):
    success: bool = True
    query: str
    requested: int
    returned: int
    items: list[MacroItem]


class MemoryItem(BaseModel):
    symbol: str
    action: str
    timestamp: str | None = None
    reasoning: str
    outcome_pnl: float | None = None
    freshness_seconds: int | None = None


class MemoryResponse(BaseModel):
    success: bool = True
    query: str
    symbol_filter: str | None = None
    requested: int
    returned: int
    items: list[MemoryItem]


class IngestionStatusResponse(BaseModel):
    success: bool = True
    chroma_path: str
    collections: dict[str, int]
    latest_timestamps: dict[str, str | None]
