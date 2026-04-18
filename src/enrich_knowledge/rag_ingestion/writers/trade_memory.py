"""Append-and-prune writer for the Chroma ``trade_memory`` collection.

Unlike macro/news, trade memory has no natural key — every recorded
outcome is a distinct event. The writer therefore always appends, and
trims the oldest rows when the collection exceeds ``max_entries`` to
match the legacy ``MemoryManager._prune_if_needed`` behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from uuid import uuid4

from src.config import StorageSettings
from src.enrich_knowledge.rag_ingestion.transforms.trade_memory import (
    TradeMemoryRecord,
)
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import _store_for

logger = logging.getLogger(__name__)

_TRADE_MEMORY_COLLECTION = "trade_memory"
_DEFAULT_MAX_ENTRIES = 500


def write_record(
    storage: StorageSettings,
    record: TradeMemoryRecord,
    *,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    id_factory: Callable[[], str] | None = None,
) -> str:
    """Append ``record``; prune oldest rows past ``max_entries``.

    Returns the generated document id so callers can log or correlate.
    ``id_factory`` overrides the uuid4 default — tests pin it to get
    deterministic ids.
    """
    store = _store_for(storage.chroma_path)
    document_id = (id_factory or (lambda: str(uuid4())))()
    store.add_document(
        _TRADE_MEMORY_COLLECTION,
        document_id=document_id,
        text=record.text,
        metadata=record.metadata,
    )
    _prune_if_needed(storage, max_entries)
    return document_id


def _prune_if_needed(storage: StorageSettings, max_entries: int) -> None:
    store = _store_for(storage.chroma_path)
    try:
        count = store.count(_TRADE_MEMORY_COLLECTION)
        if count <= max_entries:
            return
        excess = count - max_entries
        collection = store.collections[_TRADE_MEMORY_COLLECTION]
        oldest = collection.get(limit=excess, include=[])
        ids_to_delete = oldest.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.debug(
                "trade_memory pruned %d entries (was %d)",
                len(ids_to_delete),
                count,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("trade_memory pruning failed: %s", exc)
