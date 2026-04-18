"""B3: Logistic Regression Trade Outcome Predictor.

Loads models/outcome_predictor.joblib and returns win probability for a
proposed trade. Used as a hard gate in RiskManager — never shown to LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from .model_store import load

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.40  # block trade if win probability < 40%


class OutcomePredictor:
    """Predict historical win probability for a proposed trade entry."""

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._bundle: dict[str, Any] | None = load("outcome_predictor")
        self.threshold = threshold

    def predict(self, market_conditions: dict[str, Any]) -> float | None:
        """Return P(win) in [0,1] or None if model unavailable."""
        if self._bundle is None:
            return None
        try:
            model = self._bundle["model"]
            feature_cols: list[str] = self._bundle["feature_cols"]
            row = self._conditions_to_row(market_conditions, feature_cols)
            proba = model.predict_proba([row])[0][1]
            return float(proba)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomePredictor.predict failed: %s", exc)
            return None

    def should_block(
        self, market_conditions: dict[str, Any]
    ) -> tuple[bool, float | None]:
        """Return (block, win_probability). Block if probability < threshold."""
        prob = self.predict(market_conditions)
        if prob is None:
            return False, None  # model not available — don't block
        return prob < self.threshold, prob

    @staticmethod
    def _conditions_to_row(
        conditions: dict[str, Any], feature_cols: list[str]
    ) -> list[float]:
        mapping: dict[str, float] = {
            "rsi": float(conditions.get("rsi", 50)),
            "adx": float(conditions.get("adx", 20)),
            "atr_pct": float(conditions.get("atr_percentage", 1.5)),
            "trend": (
                1.0
                if conditions.get("trend_direction") == "BULLISH"
                else (-1.0 if conditions.get("trend_direction") == "BEARISH" else 0.0)
            ),
            "vol_state": 1.0 if conditions.get("volume_state") == "HIGH" else 0.0,
            "bb_pos": {"UPPER": 1.0, "MIDDLE": 0.0, "LOWER": -1.0}.get(
                str(conditions.get("bb_position", "MIDDLE")), 0.0
            ),
        }
        return [mapping.get(c, 0.0) for c in feature_cols]
