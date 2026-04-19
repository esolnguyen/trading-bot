"""Driver: Isolation Forest anomaly detector.

Trains one model per (symbol, timeframe) across every entry in
``ml_timeframes`` by shelling out to ``scripts/train_anomaly.py``.
"""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings

from ._runner import run_script

logger = logging.getLogger(__name__)


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for timeframe in settings.ml_training.ml_timeframes:
        for symbol in settings.ml_training.training_symbols:
            extra = ["--timeframe", timeframe, "--symbol", symbol.lower()]
            rc = run_script("train_anomaly.py", dry_run=dry_run, extra_args=extra)
            if rc != 0 and not dry_run:
                logger.warning(
                    "anomaly training for %s %s returned %d", symbol, timeframe, rc
                )
