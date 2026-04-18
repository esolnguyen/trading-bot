"""Business logic container for the RAG MCP server.

Owns the ChromaDB handle + small helpers that turn raw Chroma query
payloads into freshness-annotated pairs. Handlers stay thin: they call
``query_collection`` and map the pairs onto a pydantic response model.
"""

import logging
from datetime import datetime, timezone
from time import time
from typing import Any

from src.config import StorageSettings
from src.mcp_servers.rag_mcp.storage import ChromaStore

logger = logging.getLogger(__name__)


class RAGToolsService:
    """Shared state + helpers used by every RAG MCP tool."""

    COLLECTIONS = ("news", "macro", "trade_memory")

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._store: ChromaStore | None = None

    # ── store handle ─────────────────────────────────────────────────────

    def get_store(self) -> ChromaStore:
        if self._store is None:
            self._store = ChromaStore(self._settings.chroma_path)
        return self._store

    @property
    def chroma_path(self) -> str:
        return self._settings.chroma_path

    # ── querying ─────────────────────────────────────────────────────────

    def query_collection(
        self, collection: str, query: str, k: int
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ``[(document, metadata)]`` pairs for the top-k results."""
        store = self.get_store()
        if store.count(collection) == 0:
            return []
        payload = store.query(collection, query, n_results=k)
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        return list(zip(documents, metadatas))

    def count(self, collection: str) -> int:
        return self.get_store().count(collection)

    def latest_timestamp(self, collection: str) -> str | None:
        """Return the most recent ingestion timestamp (ISO) for the collection.

        Used by ``ingestion_status`` to surface staleness without the caller
        having to scan the whole collection.
        """
        store = self.get_store()
        if store.count(collection) == 0:
            return None
        try:
            raw = store.collections[collection].get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("latest_timestamp(%s) failed: %s", collection, exc)
            return None

        metadatas = raw.get("metadatas") or []
        best_epoch: int | None = None
        best_display: str | None = None
        for meta in metadatas:
            if not meta:
                continue
            ts = meta.get("published_at") or meta.get("timestamp")
            iso = self.as_iso(ts)
            if iso is None:
                continue
            # Compare via epoch for correctness across mixed formats.
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                epoch = int(dt.timestamp())
            except (ValueError, TypeError):
                continue
            if best_epoch is None or epoch > best_epoch:
                best_epoch = epoch
                best_display = iso
        return best_display

    # ── freshness ────────────────────────────────────────────────────────

    @staticmethod
    def freshness_seconds(timestamp: Any) -> int | None:
        """Return seconds since ``timestamp`` (ISO-8601 or unix epoch).

        The ingestion pipeline stores timestamps as either ISO strings
        (``macro`` collection) or unix-epoch strings/ints (``news``
        collection) — both are accepted. Returns ``None`` when the field
        is missing, zero, or unparseable.
        """
        if timestamp in (None, "", 0, "0"):
            return None

        # Numeric (int or digit-string) → unix epoch.
        if isinstance(timestamp, (int, float)):
            return max(0, int(time()) - int(timestamp))
        if isinstance(timestamp, str) and timestamp.lstrip("-").isdigit():
            try:
                return max(0, int(time()) - int(timestamp))
            except ValueError:
                return None

        # Otherwise treat as ISO-8601 (accept `Z` suffix).
        if isinstance(timestamp, str):
            try:
                text = timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(0, int(time()) - int(dt.timestamp()))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def as_iso(timestamp: Any) -> str | None:
        """Normalise a timestamp to ISO-8601 for client-side display.

        Accepts ISO strings (returned unchanged), unix epochs (int or
        numeric string), or ``None``. Returns ``None`` when unparseable.
        """
        if timestamp in (None, "", 0, "0"):
            return None
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(
                    int(timestamp), tz=timezone.utc
                ).isoformat()
            except (ValueError, OSError, OverflowError):
                return None
        if isinstance(timestamp, str):
            if timestamp.lstrip("-").isdigit():
                try:
                    return datetime.fromtimestamp(
                        int(timestamp), tz=timezone.utc
                    ).isoformat()
                except (ValueError, OSError, OverflowError):
                    return None
            return timestamp
        return None
