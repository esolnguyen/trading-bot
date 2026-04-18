"""A4: Market Cycle Classifier.

Loads the Random Forest regime model from models/regime_classifier_{timeframe}.joblib
(produced by scripts/train_regime.py) and injects the regime label into
the LLM system prompt so the LLM's behaviour adapts to market conditions.
"""

from __future__ import annotations

import logging
from typing import Any

from .model_store import load

logger = logging.getLogger(__name__)

REGIME_INSTRUCTIONS: dict[str, str] = {
    "BULL_TRENDING": (
        "MACRO REGIME: BULL MARKET (trending). "
        "Favour LONG entries on pullbacks. Be patient with profitable longs. "
        "Short positions require very strong confirmation."
    ),
    "BULL_CORRECTION": (
        "MACRO REGIME: BULL CORRECTION. Price is pulling back within a bull market. "
        "Look for LONG re-entry at strong support. "
        "Avoid chasing shorts — corrections in bull markets are often brief."
    ),
    "BEAR_TRENDING": (
        "MACRO REGIME: BEAR MARKET (trending). "
        "Do not open LONG positions. Look for SHORT entries at resistance with strong confirmation. "
        "Reduce position sizes. Tighten stops aggressively."
    ),
    "ACCUMULATION": (
        "MACRO REGIME: ACCUMULATION / RANGING. Market is range-bound after a major move. "
        "Reduce trade frequency. Only trade at clear range extremes. "
        "Widen stops to avoid noise. Favour smaller position sizes."
    ),
}


class CycleClassifier:
    """Predict macro market regime from daily OHLCV features."""

    def __init__(self, timeframe: str = "4h", symbols: list[str] | None = None) -> None:
        self._fallback: dict[str, Any] | None = load(f"regime_classifier_{timeframe}")
        self._bundles: dict[str, Any] = {}
        for sym in symbols or []:
            bundle = load(f"regime_classifier_{sym.lower()}_{timeframe}")
            if bundle is not None:
                self._bundles[sym.upper()] = bundle
        self._bundle = self._fallback

    def get_bundle(self, symbol: str | None = None) -> dict[str, Any] | None:
        if symbol:
            return self._bundles.get(symbol.upper(), self._fallback)
        return self._fallback

    def predict(
        self, daily_features: dict[str, float], symbol: str | None = None
    ) -> tuple[str, float] | None:
        """Return (regime_label, confidence) or None if model unavailable."""
        bundle = self.get_bundle(symbol)
        if bundle is None:
            return None
        try:
            model = bundle["model"]
            feature_cols: list[str] = bundle["feature_cols"]
            X = [[daily_features.get(c, 0.0) for c in feature_cols]]
            label = model.predict(X)[0]
            proba = float(max(model.predict_proba(X)[0]))
            return str(label), proba
        except Exception as exc:  # noqa: BLE001
            logger.warning("CycleClassifier.predict failed: %s", exc)
            return None

    def regime_system_prompt_suffix(self, regime: str, confidence: float) -> str:
        """Return text to append to the system prompt for the given regime."""
        instruction = REGIME_INSTRUCTIONS.get(regime, "")
        if not instruction:
            return ""
        return f"\n\n{instruction} (model confidence: {confidence:.0%})"
