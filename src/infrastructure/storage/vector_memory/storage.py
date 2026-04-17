"""Storage-side helpers for VectorMemoryService.

Handles writes: experience upsert, semantic-rule upsert, metadata sanitation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Strip None values — ChromaDB rejects NoneType in metadata."""
    return {k: v for k, v in metadata.items() if v is not None}


def store_experience(
    service: Any,
    trade_id: str,
    market_context: str,
    outcome: str,
    pnl_pct: float,
    direction: str,
    confidence: str,
    reasoning: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Upsert one trade experience into the vector collection."""
    if not service._ensure_initialized():
        service.logger.warning(
            "VectorMemoryService not initialized, cannot store experience."
        )
        return False

    try:
        document = (
            f"{direction} trade. Market: {market_context}. "
            f"Result: {outcome} ({pnl_pct:+.2f}%). "
            f"Confidence: {confidence}. Reasoning: {reasoning}"
        )
        embedding = service._embedding_model.encode(document).tolist()

        trade_metadata = {
            "outcome": outcome,
            "pnl_pct": pnl_pct,
            "direction": direction,
            "confidence": confidence,
            "market_context": market_context,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            market_regime = metadata.pop("market_regime", "NEUTRAL")
            trade_metadata["market_regime"] = market_regime
            trade_metadata.update(metadata)

        trade_metadata = sanitize_metadata(trade_metadata)

        service._collection.upsert(
            ids=[trade_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[trade_metadata],
        )

        service.logger.info(
            "Stored experience: %s (%s, %s%%)",
            trade_id, outcome, f"{pnl_pct:+.2f}",
        )
        return True

    except Exception as e:
        service.logger.error("Failed to store experience: %s", e)
        return False


def store_semantic_rule(
    service: Any,
    rule_id: str,
    rule_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Upsert a learned semantic rule into the rules collection."""
    if not service._ensure_initialized():
        return False

    try:
        embedding = service._embedding_model.encode(rule_text).tolist()
        rule_meta: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active": True,
        }
        if metadata:
            rule_meta.update(metadata)

        service._semantic_rules_collection.upsert(
            ids=[rule_id],
            embeddings=[embedding],
            documents=[rule_text],
            metadatas=[rule_meta],
        )

        service.logger.info("Stored semantic rule: %s", rule_id)
        return True

    except Exception as e:
        service.logger.error("Failed to store semantic rule: %s", e)
        return False
