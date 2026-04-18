"""Lazy-init ``SentimentScorer`` for the news ingestion job.

FinBERT loads ~500MB of model weights on first call. We cache one
scorer per process and share it across cycles. Wrapped behind an
``asyncio.Lock`` so a burst of job runs doesn't trigger concurrent
loads.
"""

from __future__ import annotations

import asyncio

from src.mcp_servers.ml_mcp.services.sentiment_scorer import SentimentScorer

_scorer: SentimentScorer | None = None
_lock = asyncio.Lock()


async def get_scorer() -> SentimentScorer:
    """Return the process-wide ``SentimentScorer``; lazy-init on first call."""
    global _scorer
    async with _lock:
        if _scorer is None:
            _scorer = SentimentScorer()
        return _scorer
