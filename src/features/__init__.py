"""Shared feature engineering — one definition for trainer and serving.

Each ML model computes its inputs in two places: offline (``ml_training/*``)
and live (``ml_mcp`` services). Keeping those in lock-step is what prevents
train/serve skew, so the canonical computations live here and both sides
import from this package rather than re-deriving the features.
"""

from __future__ import annotations

from .microstructure import (
    ANOMALY_FEATURE_COLS,
    anomaly_features_frame,
    anomaly_features_from_candles,
)
from .obv import (
    OBV_SLOPE_PERIOD,
    normalized_slope,
    obv_series,
    rolling_normalized_obv_slope,
)
from .outcome import (
    OUTCOME_FEATURE_COLS,
    bb_position_label,
    bucket_bb_pos,
    bucket_trend,
    bucket_vol_state,
    outcome_row_from_conditions,
    trend_label,
    vol_state_label,
)
from .regime import (
    REGIME_FEATURE_COLS,
    compute_regime_features,
    regime_features_from_candles,
)

__all__ = [
    "ANOMALY_FEATURE_COLS",
    "anomaly_features_frame",
    "anomaly_features_from_candles",
    "OBV_SLOPE_PERIOD",
    "obv_series",
    "normalized_slope",
    "rolling_normalized_obv_slope",
    "OUTCOME_FEATURE_COLS",
    "bucket_trend",
    "bucket_vol_state",
    "bucket_bb_pos",
    "trend_label",
    "vol_state_label",
    "bb_position_label",
    "outcome_row_from_conditions",
    "REGIME_FEATURE_COLS",
    "compute_regime_features",
    "regime_features_from_candles",
]
