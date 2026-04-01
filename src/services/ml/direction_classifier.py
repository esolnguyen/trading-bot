"""B2: XGBoost Direction Classifier.

Loads models/xgboost_direction_{timeframe}.joblib and returns a bullish probability
for the current indicator snapshot. The score is fed into the LLM prompt
as one quantitative voice — not used as an override.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from src.infrastructure.ml.model_store import load

logger = logging.getLogger(__name__)


class DirectionClassifier:
    """Predict bullish probability from a current indicator snapshot."""

    def __init__(self, timeframe: str = "4h") -> None:
        self._bundle: dict[str, Any] | None = load(f"xgboost_direction_{timeframe}")
        # Rolling history of (rsi_14, macd_hist, adx) for lag features — filled on each call
        self._history: deque[tuple[float, float, float]] = deque(maxlen=3)

    def predict_proba(self, indicators: Any, price: float | None = None) -> float | None:
        """Return P(bullish) in [0,1] or None if model unavailable."""
        if self._bundle is None:
            return None
        try:
            model = self._bundle["model"]
            feature_cols: list[str] = self._bundle["feature_cols"]
            row = self._indicators_to_row(indicators, feature_cols, price)
            if row is None:
                return None
            proba = model.predict_proba([row])[0][1]
            # Update history AFTER prediction so lags reflect previous candles
            self._history.append((
                float(getattr(indicators, "rsi_14", 50.0)),
                float(getattr(indicators, "macd_hist", 0.0)),
                float(getattr(indicators, "adx", 20.0)),
            ))
            return float(proba)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DirectionClassifier.predict_proba failed: %s", exc)
            return None

    def format_context(self, indicators: Any, price: float | None = None) -> str | None:
        """Return a formatted string for the LLM prompt or None."""
        prob = self.predict_proba(indicators, price)
        if prob is None:
            return None
        direction = "bullish" if prob >= 0.5 else "bearish"
        conviction = "high" if abs(prob - 0.5) > 0.2 else "moderate" if abs(prob - 0.5) > 0.1 else "weak"
        return (
            f"## Quantitative Signal (XGBoost)\n"
            f"- Direction model: {prob:.0%} {direction} ({conviction} conviction)\n"
            f"  → Use as baseline. If RAG news strongly contradicts, weight news heavily."
        )

    def _indicators_to_row(self, indicators: Any, feature_cols: list[str], price: float | None) -> list[float] | None:
        """Map IndicatorSet fields to the feature vector used at training time."""
        mapping: dict[str, float] = {}
        for attr in ("rsi_14", "macd_line", "macd_signal", "macd_hist",
                     "bb_upper", "bb_mid", "bb_lower", "ema_20", "ema_50",
                     "volume_sma_20", "atr", "adx", "obv_slope", "choppiness"):
            mapping[attr] = float(getattr(indicators, attr, 0.0))

        # vol_ratio: now populated from IndicatorSet directly
        mapping["vol_ratio"] = float(getattr(indicators, "vol_ratio", 1.0))

        # Derived features
        mapping["ema_spread"] = mapping["ema_20"] - mapping["ema_50"]
        mapping["atr_pct"] = mapping["atr"] / price * 100 if price and price > 0 else (
            mapping["atr"] / mapping["bb_mid"] * 100 if mapping["bb_mid"] > 0 else 0.0
        )
        bb_width = mapping["bb_upper"] - mapping["bb_lower"]
        if price is not None and bb_width > 0:
            mapping["bb_pos"] = (price - mapping["bb_lower"]) / bb_width
        else:
            mapping["bb_pos"] = 0.5  # neutral fallback

        # Lag features — use history deque; fall back to current value if history not yet full
        hist = list(self._history)  # oldest → newest, up to 3 entries
        cur_rsi  = mapping["rsi_14"]
        cur_macd = mapping["macd_hist"]
        cur_adx  = mapping["adx"]

        mapping["rsi_14_lag1"]    = hist[-1][0] if len(hist) >= 1 else cur_rsi
        mapping["rsi_14_lag2"]    = hist[-2][0] if len(hist) >= 2 else cur_rsi
        mapping["rsi_14_lag3"]    = hist[-3][0] if len(hist) >= 3 else cur_rsi
        mapping["macd_hist_lag1"] = hist[-1][1] if len(hist) >= 1 else cur_macd
        mapping["macd_hist_lag2"] = hist[-2][1] if len(hist) >= 2 else cur_macd
        mapping["macd_hist_lag3"] = hist[-3][1] if len(hist) >= 3 else cur_macd
        mapping["adx_lag1"]       = hist[-1][2] if len(hist) >= 1 else cur_adx
        mapping["adx_lag2"]       = hist[-2][2] if len(hist) >= 2 else cur_adx
        mapping["adx_lag3"]       = hist[-3][2] if len(hist) >= 3 else cur_adx

        try:
            return [mapping.get(c, 0.0) for c in feature_cols]
        except Exception:  # noqa: BLE001
            return None
