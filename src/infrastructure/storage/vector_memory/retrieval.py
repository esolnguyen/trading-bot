"""Retrieval-side helpers for VectorMemoryService.

Handles reads: similarity search over trades and rules, plus recency decay.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.domain.analysis.stats import VectorSearchResult

DEFAULT_DECAY_HALF_LIFE_DAYS = 90


def calculate_recency_score(
    trade_timestamp: str,
    half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay weight for a trade timestamp."""
    try:
        trade_dt = datetime.fromisoformat(trade_timestamp)
        if trade_dt.tzinfo is None:
            trade_dt = trade_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - trade_dt).days
        decay_rate = math.log(2) / half_life_days
        return math.exp(-decay_rate * age_days)
    except (ValueError, TypeError):
        return 0.5


def retrieve_similar_experiences(
    service: Any,
    current_context: str,
    k: int = 5,
    use_decay: bool = True,
    decay_half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS,
    where: Optional[Dict[str, Any]] = None,
) -> List[VectorSearchResult]:
    """Hybrid (similarity × recency) nearest-neighbor retrieval."""
    if not service._ensure_initialized():
        return []

    try:
        if service._collection.count() == 0:
            return []

        query_embedding = service._embedding_model.encode(current_context).tolist()

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, service._collection.count()),
        }
        if where:
            query_kwargs["where"] = where
        results = service._collection.query(**query_kwargs)

        experiences: List[VectorSearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                similarity = (
                    1 - results["distances"][0][i] if results["distances"] else 0
                )
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                if use_decay:
                    timestamp = meta.get("timestamp", "")
                    recency = calculate_recency_score(timestamp, decay_half_life_days)
                    hybrid_score = similarity * 0.7 + recency * 0.3
                else:
                    recency = 1.0
                    hybrid_score = similarity

                experiences.append(VectorSearchResult(
                    id=doc_id,
                    document=results["documents"][0][i] if results["documents"] else "",
                    similarity=round(similarity * 100, 1),
                    recency=round(recency * 100, 1),
                    hybrid_score=round(hybrid_score * 100, 1),
                    metadata=meta,
                ))

        if use_decay:
            experiences.sort(key=lambda x: x.hybrid_score, reverse=True)
            experiences = experiences[:k]

        return experiences

    except Exception as e:
        service.logger.error("Failed to retrieve experiences: %s", e)
        return []


def get_all_experiences(
    service: Any,
    limit: int = 100,
    where: Optional[Dict[str, Any]] = None,
) -> List[VectorSearchResult]:
    """Bulk fetch (no similarity scoring) — defaults to excluding UPDATE entries."""
    if not service._ensure_initialized():
        return []

    try:
        query_where = where if where else {"outcome": {"$ne": "UPDATE"}}
        results = service._collection.get(
            where=query_where,
            limit=limit,
            include=["metadatas", "documents"],
        )

        experiences: List[VectorSearchResult] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                doc = results["documents"][i] if results["documents"] else ""
                experiences.append(VectorSearchResult(
                    id=doc_id,
                    document=doc,
                    similarity=0,
                    recency=0,
                    hybrid_score=0,
                    metadata=meta,
                ))

        return experiences

    except Exception as e:
        service.logger.error("Failed to retrieve all experiences: %s", e)
        return []


def get_active_rules(service: Any, n_results: int = 5) -> List[Dict[str, Any]]:
    """Fetch up to `n_results` semantic rules currently marked active."""
    if not service._ensure_initialized():
        return []

    try:
        if service._semantic_rules_collection.count() == 0:
            return []

        all_rules = service._semantic_rules_collection.get(
            where={"active": True}, limit=n_results
        )

        rules: List[Dict[str, Any]] = []
        if all_rules and all_rules["ids"]:
            for i, rule_id in enumerate(all_rules["ids"]):
                rules.append({
                    "rule_id": rule_id,
                    "text": all_rules["documents"][i] if all_rules["documents"] else "",
                    "metadata": all_rules["metadatas"][i] if all_rules["metadatas"] else {},
                })
        return rules

    except Exception as e:
        service.logger.error("Failed to get active rules: %s", e)
        return []


def get_relevant_rules(
    service: Any,
    current_context: str,
    n_results: int = 3,
    min_similarity: float = 0.4,
) -> List[Dict[str, Any]]:
    """Semantic-match active rules to the given context; drop below threshold."""
    if not service._ensure_initialized():
        return []

    try:
        count = service._semantic_rules_collection.count()
        if count == 0:
            return []

        query_embedding = service._embedding_model.encode(current_context).tolist()

        results = service._semantic_rules_collection.query(
            query_embeddings=[query_embedding],
            where={"active": True},
            n_results=min(n_results * 2, count),
        )

        rules: List[Dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for i, rule_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1 - distance
                if similarity < min_similarity:
                    continue
                rules.append({
                    "rule_id": rule_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "similarity": round(similarity * 100, 1),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        rules.sort(key=lambda r: r["similarity"], reverse=True)
        return rules[:n_results]

    except Exception as e:
        service.logger.error("Failed to get relevant rules: %s", e)
        return []


def get_trade_metadatas(service: Any, exclude_updates: bool = True) -> List[Dict[str, Any]]:
    """Return all stored metadatas, optionally filtering out UPDATE outcomes."""
    if not service._ensure_initialized():
        return []

    all_experiences = service._collection.get(include=["metadatas"])
    if (
        not all_experiences
        or not all_experiences["ids"]
        or not all_experiences["metadatas"]
    ):
        return []

    metas = all_experiences["metadatas"]
    if exclude_updates:
        return [m for m in metas if m.get("outcome") != "UPDATE"]
    return metas
