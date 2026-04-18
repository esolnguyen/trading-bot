"""OHLCV-derived macro summary source."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class OHLCVHistorySource:
    """Generate a macro summary from recent BTC/ETH market snapshots."""

    def __init__(self, aggregator: Any | None = None) -> None:
        self.aggregator = aggregator

    async def fetch(self) -> list[dict[str, Any]]:
        if self.aggregator is None:
            now = datetime.now(timezone.utc).isoformat()
            return [
                {
                    "source": "ohlcv_history",
                    "metric": "summary",
                    "value": 0,
                    "narrative": "No market aggregator configured yet for OHLCV text summary.",
                    "timestamp": now,
                }
            ]

        snapshots = await __import__("asyncio").gather(
            self.aggregator.snapshot("BTCUSDT"),
            self.aggregator.snapshot("ETHUSDT"),
        )
        narrative = " | ".join(
            f"{snapshot.symbol} price={snapshot.price:.2f} change_24h={snapshot.change_24h_pct:.2f}%"
            for snapshot in snapshots
        )
        return [
            {
                "source": "ohlcv_history",
                "metric": "summary",
                "value": 0,
                "narrative": narrative,
                "timestamp": str(max(snapshot.timestamp for snapshot in snapshots)),
            }
        ]
