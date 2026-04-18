"""CryptoCompare news ingestion job.

Reads its API key from ``IngestionSettings.cryptocompare_api_key``.
Logs + skips the cycle (rather than crashing the scheduler) when the
key is missing or the upstream returns a 429.
"""

from __future__ import annotations

import dataclasses
import logging

import requests

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.jobs._sentiment import get_scorer
from src.enrich_knowledge.rag_ingestion.sources import cryptocompare as source
from src.enrich_knowledge.rag_ingestion.transforms.news import NewsRecord, news_records
from src.enrich_knowledge.rag_ingestion.writers.chroma_news import write_records

logger = logging.getLogger(__name__)

JOB_ID = "news.cryptocompare"
SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=900, jitter_seconds=45)


async def run(settings: EnrichKnowledgeSettings) -> None:
    api_key = settings.ingestion.cryptocompare_api_key
    if not api_key:
        logger.warning(
            "cryptocompare_api_key is empty — set CRYPTOCOMPARE_API_KEY in "
            ".env to enable this job"
        )
        return

    try:
        articles = await source.fetch(api_key)
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            logger.warning("cryptocompare rate-limited (429); skipping cycle")
            return
        raise

    records = news_records(articles)
    if settings.ingestion.sentiment_scoring_enabled:
        records = await _attach_sentiment(records)
    written = write_records(settings.storage, records)
    logger.info(
        "cryptocompare cycle complete — fetched %d, relevant %d, new writes %d",
        len(articles),
        len(records),
        written,
    )


async def _attach_sentiment(records: list[NewsRecord]) -> list[NewsRecord]:
    """Score each record's text and fold the result into metadata.

    Returns the input unchanged when FinBERT is unavailable at runtime,
    so offline / CI environments stay functional.
    """
    scorer = await get_scorer()
    scored: list[NewsRecord] = []
    for record in records:
        score = scorer.score(record.text)
        if score is None:
            scored.append(record)
            continue
        metadata = {**record.metadata, "sentiment_score": str(score)}
        scored.append(dataclasses.replace(record, metadata=metadata))
    return scored
