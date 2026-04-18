"""Pure transform for the ``trade_memory`` Chroma collection.

Trade-memory writes happen from the trading loop (not a scheduled
job), so the transform layer decouples the record shape from
``legacy.domain.trading.TradeOutcome``. Callers pass typed fields to
``trade_memory_record`` — enrich_knowledge never imports trading-loop
code, and the reader-side contract (symbol, action, reasoning,
timestamp, outcome_pnl) is enforced here on the write side instead
of relying on caller discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

MetaValue = str | int | float | bool


@dataclass(frozen=True)
class TradeMemoryRecord:
    """Write-ready record for the Chroma ``trade_memory`` collection."""

    text: str
    metadata: dict[str, MetaValue]


def trade_memory_record(
    *,
    symbol: str,
    action: str,
    reasoning: str,
    timestamp: str,
    quantity: float,
    price: float | None = None,
    order_type: str = "MARKET",
    pnl_usdt: float | None = None,
    outcome_pnl: float | None = None,
    dry_run: bool = False,
) -> TradeMemoryRecord:
    """Build a ``TradeMemoryRecord`` with the reader-expected metadata shape.

    The MCP RAG ``retrieve_memory`` handler keys off ``symbol``,
    ``action``, ``reasoning``, ``timestamp``, and ``outcome_pnl``; the
    remaining fields mirror the legacy ``MemoryManager`` for continuity
    with older trade-memory rows. ``outcome_pnl`` defaults to
    ``pnl_usdt`` when the caller doesn't override it (legacy behaviour).
    ``None`` pnl values are stored as empty strings so ChromaDB's
    metadata types remain stable across rows.
    """
    text = f"{symbol} {action} reasoning: {reasoning}"
    resolved_outcome = outcome_pnl if outcome_pnl is not None else pnl_usdt
    metadata: dict[str, MetaValue] = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": "" if price is None else price,
        "order_type": order_type,
        "dry_run": dry_run,
        "pnl_usdt": "" if pnl_usdt is None else pnl_usdt,
        "outcome_pnl": "" if resolved_outcome is None else resolved_outcome,
        "timestamp": timestamp,
        "reasoning": reasoning,
    }
    return TradeMemoryRecord(text=text, metadata=metadata)
