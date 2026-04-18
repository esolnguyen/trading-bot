"""Idempotent writer for the Chroma ``news`` collection.

Mirrors ``chroma_macro`` — different collection, same natural-key
dedupe contract. Shares the cached ``ChromaStore`` handle via the
``_store_for`` helper so both writers reuse one client per path.
"""

from __future__ import annotations

import logging

from src.config import StorageSettings
from src.enrich_knowledge.rag_ingestion.transforms.news import NewsRecord
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import _store_for

logger = logging.getLogger(__name__)

_NEWS_COLLECTION = "news"


def write_records(
    storage: StorageSettings,
    records: list[NewsRecord],
    *,
    dry_run: bool = False,
) -> int:
    """Upsert ``records`` into the news collection; return new-write count."""
    if not records:
        return 0

    store = _store_for(storage.chroma_path)
    written = 0
    for record in records:
        if store.exists(_NEWS_COLLECTION, record.document_id):
            logger.debug("skip existing %s", record.document_id)
            continue
        if dry_run:
            logger.info("[dry-run] would write %s", record.document_id)
            written += 1
            continue
        store.add_document(
            _NEWS_COLLECTION,
            document_id=record.document_id,
            text=record.text,
            metadata=record.metadata,
        )
        logger.info("wrote news record %s", record.document_id)
        written += 1
    return written
