"""B2: XGBoost Direction Classifier.

Loads models/xgboost_direction_{timeframe}.joblib and returns a bullish probability
for the current indicator snapshot. The score is fed into the LLM prompt
as one quantitative voice — not used as an override.
"""

from __future__ import annotations

import logging
from typing import Any

from src.infrastructure.ml.model_store import load

logger = logging.getLogger(__name__)


class DirectionClassifier:
    """Predict bullish probability from a current indicator snapshot."""

    def __init__(self, timeframe: str = "4h") -> None:
        self._bundle: dict[str, Any] | None = load(f"xgboost_direction_{timeframe}")

    def predict_proba(self, indicators: Any) -> float | None:
        """Return P(bullish) in [0,1] or None if model unavailable."""
        if self._bundle is None:
            return None
        try:
            model = self._bundle["model"]
            feature_cols: list[str] = self._bundle["feature_cols"]
            row = self._indicators_to_row(indicators, feature_cols)
            if row is None:
                return None
            proba = model.predict_proba([row])[0][1]
            return float(proba)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DirectionClassifier.predict_proba failed: %s", exc)
            return None

    def format_context(self, indicators: Any) -> str | None:
        """Return a formatted string for the LLM prompt or None."""
        prob = self.predict_proba(indicators)
        if prob is None:
            return None
        direction = "bullish" if prob >= 0.5 else "bearish"
        conviction = "high" if abs(prob - 0.5) > 0.2 else "moderate" if abs(prob - 0.5) > 0.1 else "weak"
        return (
            f"## Quantitative Signal (XGBoost)\n"
            f"- Direction model: {prob:.0%} {direction} ({conviction} conviction)\n"
            f"  → Use as baseline. If RAG news strongly contradicts, weight news heavily."
        )

    @staticmethod
    def _indicators_to_row(indicators: Any, feature_cols: list[str]) -> list[float] | None:
        """Map IndicatorSet fields to the feature vector used at training time."""
        mapping: dict[str, float] = {}
        for attr in ("rsi_14", "macd_line", "macd_signal", "macd_hist",
                     "bb_upper", "bb_mid", "bb_lower", "ema_20", "ema_50",
                     "volume_sma_20", "atr", "adx", "obv_slope", "choppiness"):
            mapping[attr] = float(getattr(indicators, attr, 0.0))

        # Derived features expected by training script
        mapping["ema_spread"] = mapping["ema_20"] - mapping["ema_50"]
        mapping["atr_pct"] = mapping["atr"] / mapping["bb_mid"] * 100 if mapping["bb_mid"] > 0 else 0.0
        mapping["bb_pos"] = (
            (mapping["bb_mid"] - mapping["bb_lower"]) /
            (mapping["bb_upper"] - mapping["bb_lower"])
            if (mapping["bb_upper"] - mapping["bb_lower"]) > 0 else 0.5
        )

        try:
            return [mapping.get(c, 0.0) for c in feature_cols]
        except Exception:  # noqa: BLE001
            return None
