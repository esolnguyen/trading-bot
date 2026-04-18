"""OHLCV-derived macro narrative — placeholder layer.

Legacy behaviour when no aggregator is wired: return a stub string
so the pipeline has shape-complete output. When ``MarketAggregator``
is plugged into enrich_knowledge in a later phase, ``fetch`` will
pull live BTC/ETH snapshots and generate a real narrative — the
transforms/writer layers stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OHLCVSummary:
    """Text summary of recent market activity.

    ``timestamp`` is the ISO UTC moment the summary was produced; it
    doubles as the natural-key bucket in the transforms layer.
    """

    narrative: str
    timestamp: str


async def fetch(
    aggregator: Any | None = None, *, now: datetime | None = None
) -> OHLCVSummary | None:
    """Produce an OHLCV summary.

    When ``aggregator`` is ``None`` (Phase 1 default), returns the
    legacy placeholder. When wired, this will gather BTC/ETH
    snapshots via the aggregator and format a human-readable line.
    """
    stamp = (now or datetime.now(timezone.utc)).isoformat()

    if aggregator is None:
        return OHLCVSummary(
            narrative=(
                "No market aggregator configured yet for OHLCV text summary."
            ),
            timestamp=stamp,
        )

    snapshots = [
        await aggregator.snapshot("BTCUSDT"),
        await aggregator.snapshot("ETHUSDT"),
    ]
    narrative = " | ".join(
        f"{snap.symbol} price={snap.price:.2f} "
        f"change_24h={snap.change_24h_pct:.2f}%"
        for snap in snapshots
    )
    latest_ts = max(int(snap.timestamp) for snap in snapshots)
    return OHLCVSummary(narrative=narrative, timestamp=str(latest_ts))
