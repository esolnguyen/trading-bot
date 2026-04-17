"""Vector memory service for trading experiences using ChromaDB.

Provides semantic search over historical trades to find relevant past
experiences for context-aware decision making.  The class is a thin
facade — the real work lives in sibling modules:

  * :mod:`vector_memory_storage`   — writes (experiences + semantic rules)
  * :mod:`vector_memory_retrieval` — similarity search + recency decay
  * :mod:`vector_memory_prompts`   — prompt-context formatting
  * :mod:`vector_memory_stats`     — win-rate / threshold computations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.infrastructure.storage import (
    vector_memory_prompts as _prompts,
    vector_memory_retrieval as _retrieval,
    vector_memory_stats as _stats,
    vector_memory_storage as _storage,
)


class VectorMemoryService:
    """Service for storing and retrieving trading experiences via vector similarity."""

    COLLECTION_NAME = "trading_experiences"
    SEMANTIC_RULES_COLLECTION = "semantic_rules"
    DEFAULT_DECAY_HALF_LIFE_DAYS = _retrieval.DEFAULT_DECAY_HALF_LIFE_DAYS
    RR_THRESHOLDS = _stats.RR_THRESHOLDS
    FACTOR_BUCKETS = _stats.FACTOR_BUCKETS
    FACTOR_NAMES = _stats.FACTOR_NAMES

    def __init__(
        self,
        chroma_store: Any,
        embedding_model: Any = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._client = chroma_store.get_raw_client()
        self._collection: Optional[Any] = None
        self._semantic_rules_collection: Optional[Any] = None
        self._embedding_model = embedding_model
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy-create ChromaDB collections on first use."""
        if self._initialized:
            return True
        try:
            self.logger.info("Setting up VectorMemoryService collections...")
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._semantic_rules_collection = self._client.get_or_create_collection(
                name=self.SEMANTIC_RULES_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            self.logger.info(
                "VectorMemoryService collections ready: %s experiences stored",
                self._collection.count(),
            )
            return True
        except ImportError as e:
            self.logger.warning(
                "VectorMemoryService unavailable (missing dependency): %s", e
            )
            return False
        except Exception as e:
            self.logger.error(
                "Failed to initialize VectorMemoryService: %s", e, exc_info=True
            )
            return False

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return _storage.sanitize_metadata(metadata)

    def store_experience(
        self,
        trade_id: str,
        market_context: str,
        outcome: str,
        pnl_pct: float,
        direction: str,
        confidence: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return _storage.store_experience(
            self, trade_id, market_context, outcome, pnl_pct,
            direction, confidence, reasoning, metadata,
        )

    def store_semantic_rule(
        self,
        rule_id: str,
        rule_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return _storage.store_semantic_rule(self, rule_id, rule_text, metadata)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _calculate_recency_score(
        self,
        trade_timestamp: str,
        half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS,
    ) -> float:
        return _retrieval.calculate_recency_score(trade_timestamp, half_life_days)

    def retrieve_similar_experiences(
        self,
        current_context: str,
        k: int = 5,
        use_decay: bool = True,
        decay_half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return _retrieval.retrieve_similar_experiences(
            self, current_context, k, use_decay, decay_half_life_days, where,
        )

    def get_all_experiences(
        self, limit: int = 100, where: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        return _retrieval.get_all_experiences(self, limit, where)

    def get_active_rules(self, n_results: int = 5) -> List[Dict[str, Any]]:
        return _retrieval.get_active_rules(self, n_results)

    def get_relevant_rules(
        self,
        current_context: str,
        n_results: int = 3,
        min_similarity: float = 0.4,
    ) -> List[Dict[str, Any]]:
        return _retrieval.get_relevant_rules(
            self, current_context, n_results, min_similarity
        )

    def _get_trade_metadatas(self, exclude_updates: bool = True) -> List[Dict[str, Any]]:
        return _retrieval.get_trade_metadatas(self, exclude_updates)

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------
    def get_context_for_prompt(self, current_context: str, k: int = 5) -> str:
        return _prompts.format_context_for_prompt(self, current_context, k)

    def _generate_synthetic_insight(self, meta: Dict[str, Any]) -> str:
        return _prompts.generate_synthetic_insight(meta)

    def get_stats_for_context(self, current_context: str, k: int = 20) -> Dict[str, Any]:
        return _prompts.compute_stats_for_context(self, current_context, k)

    def get_anti_patterns_for_prompt(self, k: int = 3) -> str:
        return _prompts.format_anti_patterns(self, k)

    # ------------------------------------------------------------------
    # Stats / threshold learning
    # ------------------------------------------------------------------
    def get_direction_bias(self) -> Optional[Dict[str, Any]]:
        return _stats.compute_direction_bias(self)

    def compute_confidence_stats(self) -> Dict[str, Dict[str, Any]]:
        return _stats.compute_confidence_stats(self)

    def compute_adx_performance(self) -> Dict[str, Dict[str, Any]]:
        return _stats.compute_adx_performance(self)

    def compute_factor_performance(self) -> Dict[str, Dict[str, Any]]:
        return _stats.compute_factor_performance(self)

    def compute_optimal_thresholds(self, min_sample_size: int = 5) -> Dict[str, Any]:
        return _stats.compute_optimal_thresholds(self, min_sample_size)

    def get_confidence_recommendation(
        self, min_sample_size: int = 5
    ) -> Optional[str]:
        return _stats.get_confidence_recommendation(self, min_sample_size)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    @property
    def experience_count(self) -> int:
        """Total entries including UPDATE rows."""
        if not self._ensure_initialized():
            return 0
        return self._collection.count()

    @property
    def trade_count(self) -> int:
        """Only WIN/LOSS outcomes (excludes UPDATE rows)."""
        if not self._ensure_initialized():
            return 0
        try:
            results = self._collection.get(where={"outcome": {"$ne": "UPDATE"}})
            return len(results["ids"]) if results and results["ids"] else 0
        except Exception:
            return self._collection.count()

    @property
    def semantic_rule_count(self) -> int:
        if not self._ensure_initialized():
            return 0
        return self._semantic_rules_collection.count()
