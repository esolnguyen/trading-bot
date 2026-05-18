"""enrich_knowledge: training-registry shape + default settings."""

from __future__ import annotations

from src.enrich_knowledge.config.ml_training import MLTrainingSettings
from src.enrich_knowledge.runners.run_training import MODEL_REGISTRY


EXPECTED_MODEL_KEYS = {
    "anomaly",
    "direction",
    "key_levels",
    "outcome",
    "regime",
    "retrain_all",
}


def test_model_registry_keys() -> None:
    assert set(MODEL_REGISTRY) == EXPECTED_MODEL_KEYS


def test_model_registry_callables() -> None:
    for name, fn in MODEL_REGISTRY.items():
        assert callable(fn), f"{name} is not callable"


def test_ml_training_defaults(clean_env) -> None:
    s = MLTrainingSettings()
    assert s.ml_timeframes == ["15m", "1h", "4h", "1d"]
    assert s.training_symbols == ["BTCUSDT", "ETHUSDT"]
    assert s.candle_limit == 200
