"""Persist trade outcomes into trade memory."""

from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import uuid4

from src.domain.trading import Action, TradeOutcome
from src.infrastructure.storage import ChromaStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Record meaningful trade outcomes into the trade_memory collection."""

    def __init__(
        self,
        store: ChromaStore,
        *,
        id_factory: Callable[[], str] | None = None,
        settings: Any | None = None,
    ) -> None:
        self.store = store
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._max_entries: int = getattr(settings, "trade_memory_max_entries", 500) if settings else 500

    def record(self, outcome: TradeOutcome) -> None:
        decision = outcome.decision
        if decision.action == Action.HOLD and outcome.order_id is None:
            return

        document = f"{decision.symbol} {decision.action.value} reasoning: {decision.reasoning}"
        metadata = {
            "symbol": decision.symbol,
            "action": decision.action.value,
            "quantity": decision.quantity,
            "price": "" if decision.price is None else decision.price,
            "order_type": decision.order_type,
            # Fix #3: tag dry-run trades so the LLM can distinguish hypothetical
            # history from real executed trades when reading trade_memory.
            "dry_run": outcome.dry_run,
            "pnl_usdt": "" if outcome.pnl_usdt is None else outcome.pnl_usdt,
            "outcome_pnl": "" if outcome.pnl_usdt is None else outcome.pnl_usdt,
            "timestamp": outcome.timestamp,
            "reasoning": decision.reasoning,
        }
        self.store.add_document(
            "trade_memory",
            document_id=self._id_factory(),
            text=document,
            metadata=metadata,
        )
        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        try:
            count = self.store.count("trade_memory")
            if count <= self._max_entries:
                return
            excess = count - self._max_entries
            collection = self.store.collections["trade_memory"]
            oldest = collection.get(limit=excess, include=[])
            ids_to_delete = oldest.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.debug("trade_memory pruned %d entries (was %d)", len(ids_to_delete), count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("trade_memory pruning failed: %s", exc)
