"""Sub-settings + schedule for the enrich_knowledge pipeline."""

from dataclasses import dataclass

from src.config import BinanceSettings, StorageSettings

from .ingestion import IngestionSettings
from .ml_training import MLTrainingSettings


@dataclass(frozen=True)
class EnrichKnowledgeSettings:
    """Composite handle every runner loads once at startup."""

    storage: StorageSettings
    ingestion: IngestionSettings
    ml_training: MLTrainingSettings
    binance: BinanceSettings

    @classmethod
    def load(cls) -> "EnrichKnowledgeSettings":
        return cls(
            storage=StorageSettings(),
            ingestion=IngestionSettings(),
            ml_training=MLTrainingSettings(),
            binance=BinanceSettings(),
        )


__all__ = [
    "EnrichKnowledgeSettings",
    "IngestionSettings",
    "MLTrainingSettings",
]
