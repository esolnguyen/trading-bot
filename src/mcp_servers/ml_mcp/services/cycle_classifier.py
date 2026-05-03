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

# Default text reflects BTC's trend-following structure on a 30d
# forward horizon (BULL_TRENDING median +1.3%, BEAR_TRENDING median
# −0.7%). Each symbol whose forward-return distribution diverges from
# this directional alignment goes in REGIME_INSTRUCTIONS_OVERRIDES.
# See docs/ml-backtest.md (A4 section) for the empirical grounding.
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

# ETH on a 30d forward horizon mean-reverts: BEAR_TRENDING samples
# (n=741) actually preceded +5.4% mean / +1.9% median 30d return,
# while BULL_TRENDING preceded −0.2% median. Using the BTC text would
# steer the LLM the wrong way. Numbers are pooled walk-forward,
# 2019–2026, see docs/ml-backtest.md.
REGIME_INSTRUCTIONS_OVERRIDES: dict[str, dict[str, str]] = {
    "ETHUSDT": {
        "BULL_TRENDING": (
            "MACRO REGIME: BULL CONTEXT (extended). "
            "On ETH, late-trend price tends to flatten/mean-revert over the next month — "
            "do not assume continuation. Wait for fresh strength before adding LONG; "
            "rotate out of stale longs if structure weakens."
        ),
        "BULL_CORRECTION": (
            "MACRO REGIME: BULL CORRECTION (deep on ETH). "
            "Drawdowns inside ETH bull markets historically extend further than BTC's. "
            "Wait for clear support reclaim before LONG re-entry; do not catch the knife."
        ),
        "BEAR_TRENDING": (
            "MACRO REGIME: OVERSOLD-BIASED. "
            "ETH BEAR_TRENDING historically precedes a positive 30d return as price reverts to mean. "
            "Look for LONG entries at strong support with confirmation — avoid fresh SHORT entries "
            "unless structure breaks meaningfully lower."
        ),
        "ACCUMULATION": (
            "MACRO REGIME: ACCUMULATION (volatile on ETH). "
            "30d outcome distribution is wide and skews slightly negative on the median. "
            "Trade only at clear range extremes; favour small sizes and patience over directional bias."
        ),
    },
}


class CycleClassifier:
    """Predict macro market regime from daily OHLCV features."""

    def __init__(self, timeframe: str = "4h", symbols: list[str] | None = None) -> None:
        self._fallback: dict[str, Any] | None = load(f"{timeframe}/regime_classifier")
        self._bundles: dict[str, Any] = {}
        for sym in symbols or []:
            bundle = load(f"{timeframe}/regime_classifier_{sym.lower()}")
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

    def regime_system_prompt_suffix(
        self, regime: str, confidence: float, symbol: str | None = None
    ) -> str:
        """Return text to append to the system prompt for the given regime.

        Looks up a per-symbol override first (e.g. ETH mean-reverts on a
        30d horizon, so its instructions invert the BTC defaults), then
        falls back to ``REGIME_INSTRUCTIONS``.
        """
        instruction = ""
        if symbol:
            instruction = REGIME_INSTRUCTIONS_OVERRIDES.get(
                symbol.upper(), {}
            ).get(regime, "")
        if not instruction:
            instruction = REGIME_INSTRUCTIONS.get(regime, "")
        if not instruction:
            return ""
        return f"\n\n{instruction} (model confidence: {confidence:.0%})"
