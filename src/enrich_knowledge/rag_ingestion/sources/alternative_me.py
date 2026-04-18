"""Alternative.me Fear & Greed index — network layer only.

Returns the raw upstream payload, parsed into a small dataclass.
Zero knowledge of Chroma or document shape: that's the transforms
layer's job.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import requests

_FNG_URL = "https://api.alternative.me/fng/"
_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class FearGreedReading:
    """One Fear & Greed sample — latest datapoint from the upstream feed."""

    value: str
    classification: str
    timestamp: str


async def fetch(session: Any | None = None) -> FearGreedReading | None:
    """Fetch the most recent Fear & Greed index datapoint.

    ``session`` defaults to the ``requests`` module so tests can swap
    in a stub. Returns ``None`` when the upstream payload is empty
    (rare, but the API occasionally returns ``{"data": []}``).
    """
    client = session or requests
    response = await asyncio.to_thread(client.get, _FNG_URL, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        return None
    row = rows[0]
    return FearGreedReading(
        value=str(row.get("value", "")),
        classification=str(row.get("value_classification", "unknown")),
        timestamp=str(row.get("timestamp", "")),
    )
