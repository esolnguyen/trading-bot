"""B4: Isolation Forest Anomaly Detector.

Loads models/isolation_forest_{timeframe}.joblib and detects unusual market
microstructure. When triggered, the trading cycle is paused — the LLM
is never consulted during an anomaly.
"""

from __future__ import annotations

import logging
from typing import Any

from src.infrastructure.ml.model_store import load

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect abnormal market conditions using Isolation Forest."""

    def __init__(self, timeframe: str = "4h") -> None:
        self._bundle: dict[str, Any] | None = load(f"isolation_forest_{timeframe}")

    def is_anomaly(self, snapshot: Any, indicators: Any) -> bool:
        """Return True if current market conditions are anomalous."""
        if self._bundle is None:
            return False
        try:
            model = self._bundle["model"]
            feature_cols: list[str] = self._bundle["feature_cols"]
            row = self._build_row(snapshot, indicators, feature_cols)
            prediction = model.predict([row])[0]  # -1 = anomaly, 1 = normal
            if prediction == -1:
                logger.warning(
                    "Anomaly detected — vol_ratio high or unusual microstructure. "
                    "Trading cycle will be skipped."
                )
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("AnomalyDetector.is_anomaly failed: %s", exc)
            return False

    @staticmethod
    def _build_row(snapshot: Any, indicators: Any, feature_cols: list[str]) -> list[float]:
        candles = getattr(snapshot, "candles", []) or []
        vol_ratio = 0.0
        price_vel = 0.0
        hl_range  = 0.0

        if len(candles) >= 21:
            recent_vol = float(candles[-1].volume)
            vol_sma = sum(float(c.volume) for c in candles[-21:-1]) / 20
            vol_ratio = recent_vol / vol_sma if vol_sma > 0 else 1.0

        if len(candles) >= 4:
            c0 = float(candles[-1].close)
            c3 = float(candles[-4].close)
            price_vel = (c0 - c3) / c3 * 100 if c3 > 0 else 0.0

        if candles:
            last = candles[-1]
            mid = (float(last.high) + float(last.low)) / 2
            hl_range = (float(last.high) - float(last.low)) / mid * 100 if mid > 0 else 0.0

        mapping = {
            "vol_ratio":    vol_ratio,
            "price_vel":    price_vel,
            "high_low_rng": hl_range,
        }
        return [mapping.get(c, 0.0) for c in feature_cols]
