"""CoinGecko global-market ingestion job."""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.sources import coingecko as source
from src.enrich_knowledge.rag_ingestion.transforms.macro import (
    global_market_record,
)
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import write_records

logger = logging.getLogger(__name__)

JOB_ID = "macro.coingecko"
SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=1800, jitter_seconds=60)


async def run(settings: EnrichKnowledgeSettings) -> None:
    reading = await source.fetch()
    if reading is None:
        logger.warning("coingecko returned no data; skipping cycle")
        return
    record = global_market_record(reading)
    written = write_records(settings.storage, [record])
    logger.info("coingecko cycle complete — new writes: %d", written)
