"""Canonical On-Balance-Volume slope — shared by trainer and live path.

The XGBoost direction model reads ``obv_slope`` at fit time from the
training feature builder and at inference time from ``IndicatorCalculator``.
Historically those two computed *different* quantities — the trainer used a
20-bar difference normalised by a rolling mean, while the live calculator
used a least-squares slope — a silent train/serve skew. This module is the
single definition both sides now import.
"""

from __future__ import annotations

import numpy as np

# Window length for the OBV slope. Matches IndicatorCalculator._obv_slope.
OBV_SLOPE_PERIOD = 20


def obv_series(closes, volumes) -> np.ndarray:
    """Running On-Balance Volume: +volume on up-closes, −volume on down-closes."""
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    obv = np.zeros(len(closes), dtype=float)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def normalized_slope(window) -> float:
    """Least-squares slope of ``window`` over x=0..n-1, normalised by |mean|.

    Scale-independent so the value is comparable across symbols/price levels.
    Returns 0.0 for degenerate windows (n < 2 or zero spread).
    """
    window = np.asarray(window, dtype=float)
    n = len(window)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(window.mean())
    den = float(np.sum((x - x_mean) ** 2))
    if den == 0.0:
        return 0.0
    slope = float(np.sum((x - x_mean) * (window - y_mean)) / den)
    abs_mean = abs(y_mean) if y_mean != 0.0 else 1.0
    return slope / abs_mean


def rolling_normalized_obv_slope(
    closes, volumes, period: int = OBV_SLOPE_PERIOD
) -> np.ndarray:
    """Per-bar ``normalized_slope`` over a trailing ``period`` window of OBV.

    The last element equals ``normalized_slope(obv_series(...)[-period:])`` —
    i.e. exactly what ``IndicatorCalculator._obv_slope`` returns live. Bars
    without a full window are NaN so the trainer drops them, mirroring the
    old ``diff(period)`` behaviour.
    """
    obv = obv_series(closes, volumes)
    n = len(obv)
    out = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        out[i] = normalized_slope(obv[i - period + 1 : i + 1])
    return out
