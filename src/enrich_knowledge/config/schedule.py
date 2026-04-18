"""Shared schedule contract.

Just the ``ScheduleEntry`` dataclass — the actual schedule is
assembled inside ``rag_ingestion/jobs/__init__.py`` from the SPEC
each job declares. Keeping the aggregation next to the jobs lets
every new source land via one file edit instead of two.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled ingestion job.

    ``job_id`` is the stable identifier APScheduler uses for
    coalescing + misfire handling; keep it unique and never rename
    without a cleanup of the jobstore.
    """

    job_id: str
    interval_seconds: int
    jitter_seconds: int = 30
