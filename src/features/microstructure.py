"""Canonical microstructure features for the anomaly detector.

The IsolationForest anomaly model uses three features:
``vol_ratio, price_vel, high_low_rng``. The trainer built them over a
DataFrame and the live detector over a candle list — and the volume
baseline window disagreed (trainer included the current bar, the live path
excluded it). Both now go through this module. The baseline excludes the
current bar (``shift(1)``) so a spike is measured against its *prior*
history rather than inflating its own denominator.
"""

from __future__ import annotations

import pandas as pd

ANOMALY_FEATURE_COLS = ["vol_ratio", "price_vel", "high_low_rng"]

VOL_BASELINE_PERIOD = 20  # trailing bars (excluding current) for the volume mean
PRICE_VEL_LOOKBACK = 3    # bars over which to measure % price velocity


def anomaly_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three anomaly features to ``df`` (needs close/high/low/volume)."""
    v = df["volume"]
    c = df["close"]
    h = df["high"]
    l = df["low"]

    baseline = v.shift(1).rolling(VOL_BASELINE_PERIOD).mean()
    df["vol_ratio"] = v / baseline
    df["price_vel"] = c.pct_change(PRICE_VEL_LOOKBACK) * 100
    df["high_low_rng"] = (h - l) / ((h + l) / 2) * 100
    return df


def anomaly_features_from_candles(candles: list) -> dict[str, float]:
    """Last-bar anomaly feature dict from raw candles, for live inference.

    Mirrors ``anomaly_features_frame`` at the final bar: the volume baseline
    is the mean of the ``VOL_BASELINE_PERIOD`` bars *before* the current one.
    """
    n = len(candles)

    vol_ratio = 1.0
    if n >= VOL_BASELINE_PERIOD + 1:
        recent = float(candles[-1].volume)
        prior = candles[-(VOL_BASELINE_PERIOD + 1):-1]
        base = sum(float(c.volume) for c in prior) / VOL_BASELINE_PERIOD
        vol_ratio = recent / base if base > 0 else 1.0

    price_vel = 0.0
    if n >= PRICE_VEL_LOOKBACK + 1:
        c0 = float(candles[-1].close)
        cp = float(candles[-1 - PRICE_VEL_LOOKBACK].close)
        price_vel = (c0 - cp) / cp * 100 if cp > 0 else 0.0

    high_low_rng = 0.0
    if candles:
        last = candles[-1]
        mid = (float(last.high) + float(last.low)) / 2
        high_low_rng = (float(last.high) - float(last.low)) / mid * 100 if mid > 0 else 0.0

    return {
        "vol_ratio": vol_ratio,
        "price_vel": price_vel,
        "high_low_rng": high_low_rng,
    }
