"""Driver: staleness-triggered retraining orchestrator.

Mirrors the legacy ``scripts/retrain_all.py`` logic: for each model,
check whether its on-disk artifact is older than a per-model TTL and
whether enough fresh data exists. When both pass, delegate to the
model's driver. Intended as the unit invoked by a weekly
cron/systemd timer.
"""

from __future__ import annotations

import csv
import glob
import json
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
    model_path: str
    ttl_days: float
    data_check: str  # "15m" | "1d" | "trades"
    min_rows: int


_TASKS: tuple[_Task, ...] = (
    _Task("Key Levels (DBSCAN A3)", "key_levels", key_levels.train,
          "models/key_levels_cache.json", 7.0, "1d", 100),
    _Task("Anomaly Detector (IsoForest B4)", "anomaly", anomaly.train,
          "models/isolation_forest.joblib", 7.0, "15m", 5_000),
    _Task("XGBoost Direction (B2)", "direction", direction.train,
          "models/xgboost_direction.joblib", 7.0, "15m", 5_000),
    _Task("Regime Classifier (RF A4)", "regime", regime.train,
          "models/regime_classifier.joblib", 30.0, "1d", 200),
    _Task("Outcome Predictor (LogReg B3)", "outcome", outcome.train,
          "models/outcome_predictor.joblib", 3.0, "trades", 50),
)


def _count_csv_rows(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _count_closed_trades(data_dir: str = "data") -> int:
    total = 0
    for path in glob.glob(f"{data_dir}/trade_history_*.json"):
        with open(path) as f:
            trades = json.load(f)
        if isinstance(trades, list):
            total += sum(
                1 for t in trades if str(t.get("action", "")).startswith("CLOSE")
            )
    return total


def _model_age_days(model_path: str) -> float:
    p = Path(model_path)
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 86400


def _data_ok(task: _Task) -> bool:
    if task.data_check == "trades":
        return _count_closed_trades() >= task.min_rows
    return _count_csv_rows(f"data/btcusdt_{task.data_check}.csv") >= task.min_rows


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    """Retrain every model whose artifact is stale and whose data is sufficient."""
    ran = 0
    for task in _TASKS:
        age = _model_age_days(task.model_path)
        if not _data_ok(task):
            logger.info("[SKIP] %s — insufficient data", task.name)
            continue
        if age <= task.ttl_days:
            logger.info(
                "[OK]   %s — up to date (age %.1fd ≤ ttl %.1fd)",
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
        logger.info("%s %d model(s).", verb, ran)
