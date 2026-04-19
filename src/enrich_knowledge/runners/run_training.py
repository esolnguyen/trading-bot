"""Batch training CLI.

One-shot driver: load settings, backfill OHLCV for every configured
``(symbol, timeframe)`` pair, then dispatch to the requested training
module(s). Never owns a scheduler — retraining cadence is enforced
externally (systemd timer, cron, GitHub Actions).

Usage:
    python -m src.enrich_knowledge.runners.run_training --model all
    python -m src.enrich_knowledge.runners.run_training --model anomaly
    python -m src.enrich_knowledge.runners.run_training --model all --dry-run
    python -m src.enrich_knowledge.runners.run_training --skip-backfill
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.ml_training import (
    anomaly,
    direction,
    key_levels,
    regime,
    retrain_all,
)
from src.enrich_knowledge.ml_training._runner import run_script

logger = logging.getLogger("enrich_knowledge.runners.training")

TrainFn = Callable[[EnrichKnowledgeSettings, bool], None]

# Keys double as the ``--model`` choices. ``retrain_all`` is the
# staleness-triggered orchestrator safe for weekly cron; individual
# drivers are for ad-hoc refits.
MODEL_REGISTRY: dict[str, TrainFn] = {
    "anomaly": anomaly.train,
    "direction": direction.train,
    "key_levels": key_levels.train,
    "regime": regime.train,
    "retrain_all": retrain_all.train,
}


def _resolve(names: list[str]) -> list[tuple[str, TrainFn]]:
    if names == ["all"]:
        return sorted(MODEL_REGISTRY.items())
    out: list[tuple[str, TrainFn]] = []
    for name in names:
        try:
            out.append((name, MODEL_REGISTRY[name]))
        except KeyError as exc:
            known = ", ".join(sorted(MODEL_REGISTRY) or ["<none yet>"])
            raise SystemExit(
                f"Unknown model {name!r}; known: {known}"
            ) from exc
    return out


def _backfill_ohlcv(settings: EnrichKnowledgeSettings, dry_run: bool) -> None:
    """Pull OHLCV for every configured (symbol, timeframe) before training.

    ``scripts/backfill_ohlcv.py`` is incremental: it appends only
    candles newer than the last row on disk, so running this on every
    invocation is cheap when the tree is already up to date.
    """
    timeframes = settings.ml_training.ml_timeframes
    symbols = settings.ml_training.training_symbols
    logger.info(
        "backfilling OHLCV for %d symbol(s) x %d timeframe(s)",
        len(symbols), len(timeframes),
    )
    for symbol in symbols:
        for timeframe in timeframes:
            extra = ["--symbol", symbol, "--interval", timeframe]
            rc = run_script("backfill_ohlcv.py", dry_run=dry_run, extra_args=extra)
            if rc != 0 and not dry_run:
                logger.warning(
                    "backfill for %s %s returned %d", symbol, timeframe, rc
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model name to train (repeatable). Use 'all' to train every "
            "registered model. Default: all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs, print the training plan, skip writing.",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Skip the OHLCV backfill step (use when CSVs are known fresh).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    requested = args.model or ["all"]
    targets = _resolve(requested)

    if not targets:
        logger.warning(
            "MODEL_REGISTRY is empty — nothing to train. Add drivers under "
            "ml_training/ and register them in run_training.MODEL_REGISTRY."
        )
        return

    settings = EnrichKnowledgeSettings.load()

    if not args.skip_backfill:
        _backfill_ohlcv(settings, args.dry_run)

    for name, fn in targets:
        logger.info("training %s (dry_run=%s)", name, args.dry_run)
        fn(settings, args.dry_run)
        logger.info("training %s done", name)


if __name__ == "__main__":
    main()
