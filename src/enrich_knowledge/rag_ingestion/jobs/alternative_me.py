"""Alternative.me Fear & Greed ingestion job.

Glues the three layers for this source:

1. ``sources.alternative_me.fetch`` — one HTTP call.
2. ``transforms.macro.fear_and_greed_record`` — raw → MacroRecord.
3. ``writers.chroma_macro.write_records`` — idempotent upsert.

The runner sees only ``JOB_ID`` / ``SCHEDULE`` / ``run`` so the job
is trivially swappable for a fake in tests.
"""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.sources import alternative_me as source
from src.enrich_knowledge.rag_ingestion.transforms.macro import (
    fear_and_greed_record,
)
from src.enrich_knowledge.rag_ingestion.writers.chroma_macro import write_records

logger = logging.getLogger(__name__)

JOB_ID = "macro.alternative_me"
# Alternative.me publishes once per day; half-hourly polling is cheap
# and safe given strict natural-key dedupe in the writer.
SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=1800, jitter_seconds=60)


async def run(settings: EnrichKnowledgeSettings) -> None:
    reading = await source.fetch()
    if reading is None:
        logger.warning("alternative_me returned no data; skipping cycle")
        return
    record = fear_and_greed_record(reading)
    written = write_records(settings.storage, [record])
    logger.info("alternative_me cycle complete — new writes: %d", written)
