"""Canonical feature bucketing for the trade-outcome predictor.

The LogisticRegression outcome gate (``OutcomePredictor``) is fed six
features: ``rsi, adx, atr_pct, trend, vol_state, bb_pos``. ``trend``,
``vol_state`` and ``bb_pos`` are *bucketed* from continuous indicators. The
trainer buckets a whole DataFrame; the live gate buckets one snapshot. If
the two use different thresholds — or, worse, a different source indicator —
the model predicts on a feature space it never saw at fit time.

These thresholds ARE the contract the persisted model was trained against,
so both the trainer (``ml_training/outcome.py``) and the live path
(``MLToolsService.build_market_conditions`` + ``OutcomePredictor``) import
them from here. Change a threshold ⇒ retrain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

OUTCOME_FEATURE_COLS = ["rsi", "adx", "atr_pct", "trend", "vol_state", "bb_pos"]

# ── canonical bucketing thresholds (single source of truth) ───────────────
TREND_ADX_MIN = 20.0   # ADX below this ⇒ NEUTRAL regardless of EMA spread
VOL_RATIO_HIGH = 1.5   # volume / 20-bar SMA above this ⇒ "HIGH" volume state
BB_POS_UPPER = 0.66    # bb_pos above this ⇒ UPPER band
BB_POS_LOWER = 0.33    # bb_pos below this ⇒ LOWER band


# ── vectorised bucketers (trainer, operate on Series/arrays) ──────────────
def bucket_trend(ema_spread, adx) -> np.ndarray:
    """(+1 bullish / −1 bearish / 0 neutral) from EMA spread sign + ADX gate."""
    ema_spread = np.asarray(ema_spread, dtype=float)
    adx = np.asarray(adx, dtype=float)
    out = np.zeros(np.broadcast(ema_spread, adx).shape, dtype=float)
    strong = adx >= TREND_ADX_MIN
    out = np.where(strong & (ema_spread > 0), 1.0, out)
    out = np.where(strong & (ema_spread < 0), -1.0, out)
    return out


def bucket_vol_state(vol_ratio) -> np.ndarray:
    """1.0 if volume ratio exceeds the HIGH threshold else 0.0."""
    return (np.asarray(vol_ratio, dtype=float) > VOL_RATIO_HIGH).astype(float)


def bucket_bb_pos(bb_pos) -> np.ndarray:
    """(+1 upper / −1 lower / 0 middle) from continuous Bollinger position."""
    bb_pos = np.asarray(bb_pos, dtype=float)
    out = np.zeros(bb_pos.shape, dtype=float)
    out = np.where(bb_pos > BB_POS_UPPER, 1.0, out)
    out = np.where(bb_pos < BB_POS_LOWER, -1.0, out)
    return out


# ── scalar label helpers (live gate, produce the display enums) ───────────
def trend_label(ema_spread: float, adx: float) -> str:
    v = float(bucket_trend(ema_spread, adx))
    return "BULLISH" if v > 0 else "BEARISH" if v < 0 else "NEUTRAL"


def vol_state_label(vol_ratio: float) -> str:
    return "HIGH" if float(bucket_vol_state(vol_ratio)) > 0 else "NORMAL"


def bb_position_label(bb_pos: float) -> str:
    v = float(bucket_bb_pos(bb_pos))
    return "UPPER" if v > 0 else "LOWER" if v < 0 else "MIDDLE"


# Inverse of the label helpers — maps the display enums back to the numeric
# feature value. Pure lookup (no thresholds), so it cannot reintroduce skew.
_TREND_NUM = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
_BB_NUM = {"UPPER": 1.0, "MIDDLE": 0.0, "LOWER": -1.0}


def outcome_row_from_conditions(
    conditions: dict[str, Any], feature_cols: list[str]
) -> list[float]:
    """Assemble the model feature row from a market-conditions dict."""
    mapping: dict[str, float] = {
        "rsi": float(conditions.get("rsi", 50)),
        "adx": float(conditions.get("adx", 20)),
        "atr_pct": float(conditions.get("atr_percentage", 1.5)),
        "trend": _TREND_NUM.get(str(conditions.get("trend_direction")), 0.0),
        "vol_state": 1.0 if conditions.get("volume_state") == "HIGH" else 0.0,
        "bb_pos": _BB_NUM.get(str(conditions.get("bb_position", "MIDDLE")), 0.0),
    }
    return [mapping.get(c, 0.0) for c in feature_cols]
