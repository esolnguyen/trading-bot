"""B1: FinBERT News Sentiment Scorer.

Scores news article text at ingestion time using ProsusAI/finbert.
Requires transformers + torch. Loaded lazily so the bot starts
even if these deps are absent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentimentScorer:
    """Score text sentiment using FinBERT (-1 bearish → +1 bullish)."""

    _MODEL_NAME = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._available: bool | None = None  # None = not yet checked

    def _load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._MODEL_NAME,
                top_k=None,          # return all three labels
                truncation=True,
                max_length=512,
            )
            self._available = True
            logger.info("FinBERT sentiment scorer loaded (%s)", self._MODEL_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.debug("FinBERT unavailable: %s — sentiment scoring disabled", exc)
            self._available = False
        return self._available  # type: ignore[return-value]

    def score(self, text: str) -> float | None:
        """Return a sentiment score in [-1, +1] or None if unavailable.

        +1 = strongly positive/bullish, -1 = strongly negative/bearish.
        """
        if not self._load():
            return None
        try:
            text = text[:512]  # FinBERT max token length guard
            results: list[dict[str, Any]] = self._pipeline(text)[0]  # type: ignore[index]
            label_scores = {r["label"].lower(): r["score"] for r in results}
            positive = label_scores.get("positive", 0.0)
            negative = label_scores.get("negative", 0.0)
            # Normalised score: +1 all positive, -1 all negative, 0 neutral
            return round(positive - negative, 4)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SentimentScorer.score failed: %s", exc)
            return None

    def score_batch(self, texts: list[str]) -> list[float | None]:
        """Score multiple texts. Falls back gracefully on failure."""
        return [self.score(t) for t in texts]
