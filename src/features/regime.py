"""Canonical daily-regime feature engineering — shared by trainer + serving.

The RandomForest macro-regime classifier is fit on these 11 features in
``ml_training/regime.py`` and queried live in ``MLToolsService``. The live
path previously rebuilt the features by hand from candle lists and even
hardcoded ``adx_14 = 0.0`` — so the model was trained with a real ADX but
served a constant zero (and a differently-computed EMA slope). Both sides
now call ``compute_regime_features`` so the served vector matches training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Candles per calendar day, per timeframe — drives the window sizing below.
CANDLES_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}

REGIME_FEATURE_COLS = [
    "ema50_dist", "ema100_dist", "ema200_dist",
    "ema50_slope", "ema200_slope",
    "high_52w_dist", "low_52w_dist",
    "adx_14", "realized_vol",
    "hh_count", "ll_count",
]


def compute_regime_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Add the regime feature columns to ``df`` (needs close/high/low)."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    cpd = CANDLES_PER_DAY[timeframe]

    for span in [50, 100, 200]:
        ema = c.ewm(span=span * cpd, adjust=False).mean()
        df[f"ema{span}_dist"] = (c - ema) / ema
        df[f"ema{span}_slope"] = ema.diff(5 * cpd) / ema

    w365 = 365 * cpd
    df["high_52w_dist"] = (c - h.rolling(w365, min_periods=w365 // 12).max()) / c
    df["low_52w_dist"] = (c - l.rolling(w365, min_periods=w365 // 12).min()) / c

    adx_com = 13 * cpd
    up = h.diff()
    down = -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=adx_com, adjust=False).mean()
    pdi = 100 * pdm.ewm(com=adx_com, adjust=False).mean() / atr14.replace(0, np.nan)
    mdi = 100 * mdm.ewm(com=adx_com, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx_14"] = dx.ewm(com=adx_com, adjust=False).mean()

    w30 = 30 * cpd
    df["realized_vol"] = c.pct_change().rolling(w30).std() * np.sqrt(365 * cpd)

    w90 = 90 * cpd
    df["hh_count"] = h.rolling(w90).apply(
        lambda x: sum(x[i] > x[i - 1] for i in range(1, len(x))), raw=True
    )
    df["ll_count"] = l.rolling(w90).apply(
        lambda x: sum(x[i] < x[i - 1] for i in range(1, len(x))), raw=True
    )

    return df


def regime_features_from_candles(candles: list, timeframe: str = "1d") -> dict[str, float]:
    """Last-bar regime feature dict from raw candles, for live inference.

    Builds a DataFrame and reuses ``compute_regime_features`` so the served
    vector is byte-for-byte the trainer's. Missing/short-window values fall
    back to 0.0 (mirrors the historical inference contract).
    """
    df = pd.DataFrame(
        {
            "close": [float(c.close) for c in candles],
            "high": [float(c.high) for c in candles],
            "low": [float(c.low) for c in candles],
        }
    )
    df = compute_regime_features(df, timeframe=timeframe)
    last: Any = df.iloc[-1]
    return {
        col: (float(last[col]) if pd.notna(last[col]) else 0.0)
        for col in REGIME_FEATURE_COLS
    }
