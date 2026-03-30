"""Analysis statistics domain models (vector memory and cost tracking)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from src.shared.data_utils import SerializableMixin


@dataclass(slots=True)
class VectorSearchResult(SerializableMixin):
    """A single result retrieved from the vector memory store."""

    id: str
    document: str
    similarity: float
    recency: float
    hybrid_score: float
    metadata: Dict[str, Any]


@dataclass(slots=True)
class ConfidenceLevelStats(SerializableMixin):
    """Performance statistics for a single confidence level (HIGH/MEDIUM/LOW)."""

    win_rate: float
    avg_pnl: float
    total_trades: int


@dataclass(slots=True)
class ADXBucketStats(SerializableMixin):
    """Performance statistics for an ADX range bucket."""

    bucket: str  # e.g. "0-20", "20-40"
    win_rate: float
    avg_pnl: float
    total_trades: int


@dataclass(slots=True)
class FactorPerformance(SerializableMixin):
    """Performance metrics for a confluence factor."""

    factor_name: str
    win_rate: float
    avg_score: float
    sample_size: int


@dataclass(slots=True)
class SemanticRule(SerializableMixin):
    """A semantic trading rule learned from win/loss clusters."""

    rule_id: str
    rule_text: str
    win_rate: Optional[float] = None
    source_trades: Optional[int] = None
    created_at: Optional[datetime] = None
    similarity: float = 0.0


@dataclass(slots=True)
class TokenUsageStats(SerializableMixin):
    """Token usage statistics from a single API request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None


@dataclass(slots=True)
class SessionCosts(SerializableMixin):
    """Cumulative session costs by provider."""

    openrouter: float = 0.0
    google: float = 0.0
    lmstudio: float = 0.0

    @property
    def total(self) -> float:
        """Total cost across all providers."""
        return self.openrouter + self.google + self.lmstudio


@dataclass(slots=True)
class ProviderCostStats(SerializableMixin):
    """Persistent cost statistics for a single provider."""

    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


__all__ = [
    "VectorSearchResult",
    "ConfidenceLevelStats",
    "ADXBucketStats",
    "FactorPerformance",
    "SemanticRule",
    "TokenUsageStats",
    "SessionCosts",
    "ProviderCostStats",
]
