"""B4: Isolation Forest Anomaly Detector.

Loads models/isolation_forest_{timeframe}.joblib and detects unusual market
microstructure. When triggered, the trading cycle is paused — the LLM
is never consulted during an anomaly.
"""

from __future__ import annotations

import logging
from typing import Any

from src.features.microstructure import (
    anomaly_features_from_candles,
)

from .model_store import load

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect abnormal market conditions using Isolation Forest."""

    def __init__(self, timeframe: str = "4h", symbols: list[str] | None = None) -> None:
        self._fallback: dict[str, Any] | None = load(f"{timeframe}/isolation_forest")
        self._bundles: dict[str, Any] = {}
        for sym in symbols or []:
            bundle = load(f"{timeframe}/isolation_forest_{sym.lower()}")
            if bundle is not None:
                self._bundles[sym.upper()] = bundle
        # Keep legacy attribute pointing at fallback for external callers
        self._bundle = self._fallback

    def _get_bundle(self, symbol: str | None) -> dict[str, Any] | None:
        if symbol:
            return self._bundles.get(symbol.upper(), self._fallback)
        return self._fallback

    def is_anomaly(self, snapshot: Any, indicators: Any) -> bool:
        """Return True if current market conditions are anomalous."""
        symbol = getattr(snapshot, "symbol", None)
        bundle = self._get_bundle(symbol)
        if bundle is None:
            return False
        try:
            model = bundle["model"]
            feature_cols: list[str] = bundle["feature_cols"]
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
    def _build_row(
        snapshot: Any, indicators: Any, feature_cols: list[str]
    ) -> list[float]:
        candles = getattr(snapshot, "candles", []) or []
        # Shared with the trainer (features.microstructure) so the volume
        # baseline window and formulas match exactly.
        mapping = anomaly_features_from_candles(candles)
        return [mapping.get(c, 0.0) for c in feature_cols]
