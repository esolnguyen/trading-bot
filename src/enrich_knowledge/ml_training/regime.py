"""Driver: Random Forest regime classifier."""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings

from ._runner import run_script

logger = logging.getLogger(__name__)

_REGIME_TIMEFRAME = "1d"


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for symbol in settings.ml_training.training_symbols:
        extra = ["--timeframe", _REGIME_TIMEFRAME, "--symbol", symbol.lower()]
        rc = run_script("train_regime.py", dry_run=dry_run, extra_args=extra)
        if rc != 0 and not dry_run:
            logger.warning(
                "regime training for %s returned %d", symbol, rc
            )
