"""DefiLlama TVL — network layer only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

_URL = "https://api.llama.fi/chains"
_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class TvlReading:
    """Aggregated TVL across all chains DefiLlama tracks.

    DefiLlama's ``/chains`` endpoint doesn't publish its own
    timestamp, so we stamp one at fetch time (``captured_at``, ISO
    UTC) — that timestamp is also the natural-key bucket.
    """

    total_tvl_usd: float
    captured_at: str


async def fetch(
    session: Any | None = None, *, now: datetime | None = None
) -> TvlReading | None:
    """Fetch and aggregate TVL across all tracked chains."""
    client = session or requests
    response = await asyncio.to_thread(client.get, _URL, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json() or []
    if not payload:
        return None
    total = sum(float(row.get("tvl", 0) or 0) for row in payload)
    captured_at = (now or datetime.now(timezone.utc)).isoformat()
    return TvlReading(total_tvl_usd=total, captured_at=captured_at)
