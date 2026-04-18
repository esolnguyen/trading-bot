"""Driver: Outcome (win/loss) Logistic Regression.

Trained once from all ``data/trade_history_*.json`` files; no
per-symbol split. Needs ≥ ``--min-trades`` closed trades on disk.
"""

from __future__ import annotations

import logging

from src.enrich_knowledge.config import EnrichKnowledgeSettings

from ._runner import run_script

logger = logging.getLogger(__name__)


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    del settings  # no settings knobs yet — legacy script uses defaults
    rc = run_script("train_outcome.py", dry_run=dry_run)
    if rc != 0 and not dry_run:
        logger.warning("outcome training returned %d", rc)
