"""OHLCV-history macro summary job.

Pulls live BTC/ETH snapshots via the shared ``MarketAggregator`` and
hands the resulting narrative to the macro writer. Feed + aggregator
are lazy-init'd once per process (see ``_aggregator.py``) — every
cycle reuses the same client.
"""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.jobs._aggregator import get_aggregator
from src.enrich_knowledge.rag_ingestion.sources import ohlcv_history as source
from src.enrich_knowledge.rag_ingestion.transforms.macro import (
    ohlcv_summary_record,
)
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import write_records

logger = logging.getLogger(__name__)

JOB_ID = "macro.ohlcv_history"
SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=3600, jitter_seconds=120)


async def run(settings: EnrichKnowledgeSettings) -> None:
    try:
        aggregator = await get_aggregator(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ohlcv_history aggregator unavailable (%s); skipping cycle", exc
        )
        return

    try:
        summary = await source.fetch(aggregator=aggregator)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ohlcv_history fetch failed (%s); skipping cycle", exc)
        return

    if summary is None:
        logger.warning("ohlcv_history returned no data; skipping cycle")
        return
    record = ohlcv_summary_record(summary)
    written = write_records(settings.storage, [record])
    logger.info("ohlcv_history cycle complete — new writes: %d", written)
