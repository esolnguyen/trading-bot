"""Driver: staleness-triggered retraining orchestrator.

For each model family, look up the newest artifact on disk (glob over
``models/<tf>/<family>*``), compare against a per-model TTL, and
delegate to the family driver when stale. The family drivers iterate
over every ``(symbol, timeframe)`` pair themselves.
"""

from __future__ import annotations

import csv
import glob
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.enrich_knowledge.config import EnrichKnowledgeSettings

from . import anomaly, direction, key_levels, outcome, regime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Task:
    name: str
    driver_name: str
    train_fn: object  # callable (settings, dry_run) -> None
    artifact_glob: str
    ttl_days: float
    data_check: str
    min_rows: int


_TASKS: tuple[_Task, ...] = (
    _Task("Key Levels (DBSCAN A3)", "key_levels", key_levels.train,
          "models/1d/key_levels*cache.json", 7.0, "1d", 100),
    _Task("Anomaly Detector (IsoForest B4)", "anomaly", anomaly.train,
          "models/*/isolation_forest*.joblib", 7.0, "15m", 5_000),
    _Task("XGBoost Direction (B2)", "direction", direction.train,
          "models/*/xgboost_direction*.joblib", 7.0, "15m", 5_000),
    _Task("Regime Classifier (RF A4)", "regime", regime.train,
          "models/1d/regime_classifier*.joblib", 30.0, "1d", 200),
    _Task("Outcome Predictor (LogReg B3)", "outcome", outcome.train,
          "models/outcome_predictor*.joblib", 7.0, "4h", 1_500),
)


def _count_csv_rows(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _newest_mtime(pattern: str) -> float | None:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(Path(p).stat().st_mtime for p in matches)


def _model_age_days(pattern: str) -> float:
    newest = _newest_mtime(pattern)
    if newest is None:
        return float("inf")
    return (time.time() - newest) / 86400


def _data_ok(task: _Task) -> bool:
    return _count_csv_rows(f"data/ohlcv/btcusdt_{task.data_check}.csv") >= task.min_rows


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    """Retrain every model family whose newest artifact is stale."""
    ran = 0
    for task in _TASKS:
        age = _model_age_days(task.artifact_glob)
        if not _data_ok(task):
            logger.info("[SKIP] %s — insufficient data", task.name)
            continue
        if age <= task.ttl_days:
            logger.info(
                "[OK]   %s — up to date (age %.1fd <= ttl %.1fd)",
                task.name, age, task.ttl_days,
            )
            continue
        logger.info(
            "[RUN]  %s — age %.1fd > ttl %.1fd", task.name, age, task.ttl_days
        )
        task.train_fn(settings, dry_run)
        ran += 1

    if ran == 0:
        logger.info("All models are up to date. Nothing retrained.")
    else:
        verb = "Would retrain" if dry_run else "Retrained"
        logger.info("%s %d model family/families.", verb, ran)
