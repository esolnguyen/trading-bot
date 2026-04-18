"""Idempotent writer for the Chroma ``macro`` collection.

Owns the natural-key dedupe contract: given a ``MacroRecord`` with a
stable ``document_id``, a second call is a no-op. ``ChromaStore`` is
imported from the MCP server package — it's the one concrete piece of
infra shared across the read/write boundary (see plan.md Phase 1).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import StorageSettings
from src.enrich_knowledge.rag_ingestion.transforms.macro import MacroRecord
from src.mcp_servers.rag_mcp.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_MACRO_COLLECTION = "macro"


@lru_cache(maxsize=4)
def _store_for(chroma_path: str) -> ChromaStore:
    """One ChromaStore per path — collections open lazily on first write."""
    return ChromaStore(chroma_path)


def write_records(
    storage: StorageSettings,
    records: list[MacroRecord],
    *,
    dry_run: bool = False,
) -> int:
    """Upsert ``records`` into the macro collection; return new-write count.

    Existing ``document_id`` hits are skipped without error (dedupe by
    natural key). ``dry_run=True`` prints the plan and writes nothing.
    """
    if not records:
        return 0

    store = _store_for(storage.chroma_path)
    written = 0
    for record in records:
        if store.exists(_MACRO_COLLECTION, record.document_id):
            logger.debug("skip existing %s", record.document_id)
            continue
        if dry_run:
            logger.info("[dry-run] would write %s", record.document_id)
            written += 1
            continue
        store.add_document(
            _MACRO_COLLECTION,
            document_id=record.document_id,
            text=record.text,
            metadata=record.metadata,
        )
        logger.info("wrote macro record %s", record.document_id)
        written += 1
    return written
