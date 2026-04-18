"""Driver: DBSCAN-based key-level cache (support/resistance)."""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings

from ._runner import run_script

logger = logging.getLogger(__name__)


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for symbol in settings.ml_training.training_symbols:
        extra = ["--symbol", symbol.lower()]
        rc = run_script("fit_key_levels.py", dry_run=dry_run, extra_args=extra)
        if rc != 0 and not dry_run:
            logger.warning(
                "key_levels fit for %s returned %d", symbol, rc
            )
