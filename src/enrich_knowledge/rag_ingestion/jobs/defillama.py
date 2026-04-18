"""DefiLlama TVL ingestion job."""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.sources import defillama as source
from src.enrich_knowledge.rag_ingestion.transforms.macro import tvl_record
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import write_records

logger = logging.getLogger(__name__)

JOB_ID = "macro.defillama"
# Matches the legacy `rag_defillama_update_interval_hours: 0.25` knob.
SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=900, jitter_seconds=45)


async def run(settings: EnrichKnowledgeSettings) -> None:
    reading = await source.fetch()
    if reading is None:
        logger.warning("defillama returned no data; skipping cycle")
        return
    record = tvl_record(reading)
    written = write_records(settings.storage, [record])
    logger.info("defillama cycle complete — new writes: %d", written)
