"""Job composition root — aggregates every scheduled ingestion job.

Each job module exposes three public symbols:

* ``JOB_ID``      — stable string id (used by APScheduler)
* ``SCHEDULE``    — ``ScheduleEntry`` with cadence + jitter
* ``run(settings)`` — async entrypoint that does fetch → transform → write

``SCHEDULES`` and ``REGISTRY`` below are the two tuples the runner
consumes. Adding a new job is: one file under ``jobs/`` + one line
here.
"""

from collections.abc import Awaitable, Callable

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.config.schedule import ScheduleEntry

from . import alternative_me, coingecko, cryptocompare, defillama, ohlcv_history

JobFn = Callable[[EnrichKnowledgeSettings], Awaitable[None]]

_JOBS = (
    alternative_me,
    coingecko,
    defillama,
    cryptocompare,
    ohlcv_history,
)

SCHEDULES: tuple[ScheduleEntry, ...] = tuple(job.SCHEDULE for job in _JOBS)
REGISTRY: dict[str, JobFn] = {job.JOB_ID: job.run for job in _JOBS}


__all__ = ["SCHEDULES", "REGISTRY", "JobFn"]
