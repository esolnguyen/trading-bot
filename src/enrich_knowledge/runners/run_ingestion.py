"""Long-running ingestion loop.

Walks the aggregated ``SCHEDULES`` from
``rag_ingestion.jobs`` and registers one APScheduler job per entry.
Each job dispatches through ``REGISTRY[job_id]`` — the runner
itself never learns about individual sources, so tests can swap in
fake jobs with no scheduler changes.

Adding a new source is: one file under ``rag_ingestion/jobs/`` + one
line in ``rag_ingestion/jobs/__init__.py``.

Run locally:
    python -m src.enrich_knowledge.runners.run_ingestion
    python -m src.enrich_knowledge.runners.run_ingestion --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry
from src.enrich_knowledge.rag_ingestion.jobs import REGISTRY, SCHEDULES, JobFn
from src.enrich_knowledge.rag_ingestion.jobs import _aggregator

logger = logging.getLogger("enrich_knowledge.runners.ingestion")


def resolve_job(job_id: str) -> JobFn:
    try:
        return REGISTRY[job_id]
    except KeyError as exc:
        raise RuntimeError(
            f"Schedule references unknown job_id {job_id!r}; "
            f"every ScheduleEntry must be present in REGISTRY"
        ) from exc


async def _run_entry(entry: ScheduleEntry, settings: EnrichKnowledgeSettings) -> None:
    """Swallow per-source failures so one bad source can't kill the loop."""
    try:
        await resolve_job(entry.job_id)(settings)
    except Exception:  # noqa: BLE001
        logger.exception("ingestion job %s failed", entry.job_id)


def _register(scheduler: AsyncIOScheduler, settings: EnrichKnowledgeSettings) -> None:
    for entry in SCHEDULES:
        scheduler.add_job(
            _run_entry,
            trigger=IntervalTrigger(
                seconds=entry.interval_seconds, jitter=entry.jitter_seconds
            ),
            args=(entry, settings),
            id=entry.job_id,
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=entry.interval_seconds,
        )
        logger.info(
            "registered %s every %ds (jitter %ds)",
            entry.job_id,
            entry.interval_seconds,
            entry.jitter_seconds,
        )


async def _dry_run(settings: EnrichKnowledgeSettings) -> None:
    """Print the plan without starting a scheduler.

    Validates the REGISTRY ↔ SCHEDULES contract without touching the
    network. Exits 1 if any schedule entry is unwired.
    """
    missing: list[str] = []
    for entry in SCHEDULES:
        if entry.job_id not in REGISTRY:
            missing.append(entry.job_id)
            continue
        logger.info(
            "[dry-run] would schedule %s every %ds",
            entry.job_id,
            entry.interval_seconds,
        )
    if missing:
        logger.error("unwired job ids: %s", ", ".join(missing))
        raise SystemExit(1)
    logger.info(
        "[dry-run] %d source(s) would be scheduled against chroma=%s",
        len(SCHEDULES),
        settings.storage.chroma_path,
    )


async def _run_forever(settings: EnrichKnowledgeSettings) -> None:
    scheduler = AsyncIOScheduler()
    _register(scheduler, settings)
    scheduler.start()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    if not SCHEDULES:
        logger.warning(
            "SCHEDULES is empty — runner is idling. Add a job under "
            "rag_ingestion/jobs/ and register it in jobs/__init__.py."
        )

    logger.info("ingestion loop started; Ctrl-C to stop")
    await stop.wait()

    logger.info("shutdown requested; stopping scheduler")
    scheduler.shutdown(wait=False)
    await _aggregator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schedule + validate job wiring, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = EnrichKnowledgeSettings.load()

    if args.dry_run:
        asyncio.run(_dry_run(settings))
        return

    asyncio.run(_run_forever(settings))


if __name__ == "__main__":
    main()
