"""Pure transforms for the ``macro`` Chroma collection.

No network, no disk. Every function takes raw upstream data and
returns a ``MacroRecord`` ready for the writer layer. The
``document_id`` is the natural key that makes re-ingestion
idempotent — collisions mean "already stored", not a duplicate to
reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.enrich_knowledge.rag_ingestion.sources.alternative_me import (
    FearGreedReading,
)
from src.enrich_knowledge.rag_ingestion.sources.coingecko import (
    GlobalMarketReading,
)
from src.enrich_knowledge.rag_ingestion.sources.defillama import TvlReading
from src.enrich_knowledge.rag_ingestion.sources.ohlcv_history import (
    OHLCVSummary,
)


@dataclass(frozen=True)
class MacroRecord:
    """Write-ready record for the Chroma ``macro`` collection."""

    document_id: str
    text: str
    metadata: dict[str, str]


def _bucket(timestamp: str) -> str:
    """Natural-key bucket — first 13 chars of a timestamp.

    Upstreams publish on different cadences (F&G daily, CoinGecko
    minute-resolution, etc.), but any sub-second precision would break
    natural-key dedupe. Fall back to ``"unknown"`` to mirror legacy
    behaviour when the timestamp is missing.
    """
    return timestamp[:13] or "unknown"


def fear_and_greed_record(reading: FearGreedReading) -> MacroRecord:
    """F&G reading → macro record.

    Document id uses the 13-char timestamp bucket; re-running within
    the same day is a no-op.
    """
    return MacroRecord(
        document_id=f"alternative_me_{_bucket(reading.timestamp)}",
        text=f"Fear and Greed Index is {reading.value} ({reading.classification}).",
        metadata={
            "source": "alternative_me",
            "metric": "fear_and_greed",
            "value": reading.value,
            "classification": reading.classification,
            "timestamp": reading.timestamp,
        },
    )


def global_market_record(reading: GlobalMarketReading) -> MacroRecord:
    """CoinGecko global market → macro record."""
    return MacroRecord(
        document_id=f"coingecko_{_bucket(reading.timestamp)}",
        text=(
            f"Global crypto market cap changed "
            f"{reading.market_cap_change_24h_pct}% in 24h."
        ),
        metadata={
            "source": "coingecko",
            "metric": "market_cap_change_percentage_24h_usd",
            "value": str(reading.market_cap_change_24h_pct),
            "timestamp": reading.timestamp,
        },
    )


def tvl_record(reading: TvlReading) -> MacroRecord:
    """DefiLlama TVL → macro record.

    DefiLlama doesn't publish a per-query timestamp, so the bucket
    uses a caller-supplied ``captured_at`` (ISO string, set at fetch
    time) to keep natural-key dedupe meaningful.
    """
    return MacroRecord(
        document_id=f"defillama_{_bucket(reading.captured_at)}",
        text=(
            f"Total tracked DeFi TVL is {reading.total_tvl_usd:.2f} USD "
            f"across reported chains."
        ),
        metadata={
            "source": "defillama",
            "metric": "total_tvl",
            "value": f"{reading.total_tvl_usd:.2f}",
            "timestamp": reading.captured_at,
        },
    )


def ohlcv_summary_record(summary: OHLCVSummary) -> MacroRecord:
    """OHLCV narrative → macro record.

    When no aggregator is wired (Phase 1 default), the summary is a
    placeholder so the pipeline still has shape-complete output.
    """
    return MacroRecord(
        document_id=f"ohlcv_history_{_bucket(summary.timestamp)}",
        text=summary.narrative,
        metadata={
            "source": "ohlcv_history",
            "metric": "summary",
            "value": "0",
            "timestamp": summary.timestamp,
        },
    )
