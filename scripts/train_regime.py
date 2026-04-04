#!/usr/bin/env python3
"""F: Train macro regime classifier (A4).

Uses daily OHLCV to classify market into BULL_TRENDING, BULL_CORRECTION,
BEAR_TRENDING, ACCUMULATION. Retrain monthly.

Usage:
    python scripts/train_regime.py
    python scripts/train_regime.py --csv data/btcusdt_1d.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report


FEATURE_COLS = [
    "ema50_dist", "ema100_dist", "ema200_dist",
    "ema50_slope", "ema200_slope",
    "high_52w_dist", "low_52w_dist",
    "adx_14", "realized_vol",
    "hh_count", "ll_count",
]


_CPD = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}  # candles per day


def compute_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    cpd = _CPD[timeframe]

    for span in [50, 100, 200]:
        ema = c.ewm(span=span * cpd, adjust=False).mean()
        df[f"ema{span}_dist"]  = (c - ema) / ema
        df[f"ema{span}_slope"] = ema.diff(5 * cpd) / ema

    w365 = 365 * cpd
    w90  = 90  * cpd
    w30  = 30  * cpd
    df["high_52w_dist"] = (c - h.rolling(w365, min_periods=w365 // 12).max()) / c
    df["low_52w_dist"]  = (c - l.rolling(w365, min_periods=w365 // 12).min()) / c

    # ADX-14 (14-day equivalent)
    adx_com = 13 * cpd
    up   = h.diff()
    down = -l.diff()
    pdm  = up.where((up > down) & (up > 0), 0.0)
    mdm  = down.where((down > up) & (down > 0), 0.0)
    prev_c = c.shift(1)
    tr   = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=adx_com, adjust=False).mean()
    pdi  = 100 * pdm.ewm(com=adx_com, adjust=False).mean() / atr14.replace(0, np.nan)
    mdi  = 100 * mdm.ewm(com=adx_com, adjust=False).mean() / atr14.replace(0, np.nan)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx_14"] = dx.ewm(com=adx_com, adjust=False).mean()

    # 30-day realised volatility (annualised)
    df["realized_vol"] = c.pct_change().rolling(w30).std() * np.sqrt(365 * cpd)

    # Higher highs / lower lows over 90-day equivalent window
    df["hh_count"] = h.rolling(w90).apply(lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))), raw=True)
    df["ll_count"] = l.rolling(w90).apply(lambda x: sum(x[i] < x[i-1] for i in range(1, len(x))), raw=True)

    return df


def label_regime(row: pd.Series) -> str:
    """Rule-based labeller — model generalises beyond these rules."""
    e50  = row["ema50_dist"]
    e200 = row["ema200_dist"]
    sl50 = row["ema50_slope"]
    h52  = row["high_52w_dist"]

    if e200 > 0.05 and sl50 > 0:
        return "BULL_TRENDING"
    if e200 > -0.05 and h52 < -0.10:
        return "BULL_CORRECTION"
    if e200 < -0.05 and sl50 < 0:
        return "BEAR_TRENDING"
    return "ACCUMULATION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1d", choices=["1d", "4h", "1h", "15m"],
                        help="Candle timeframe; auto-selects --csv and --out if not overridden.")
    parser.add_argument("--symbol", default=None, help="Symbol e.g. btcusdt — prefixes output filename")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sym = args.symbol.lower() if args.symbol else "btcusdt"
    sym_prefix = f"{sym}_" if args.symbol else ""
    if args.csv is None:
        args.csv = f"data/ohlcv/{sym}_{args.timeframe}.csv"
    if args.out is None:
        args.out = f"models/regime_classifier_{sym_prefix}{args.timeframe}.joblib"

    print(f"Loading {args.csv} …")
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded  [{args.timeframe}]")

    df = compute_features(df, timeframe=args.timeframe)
    df["label"] = df.apply(label_regime, axis=1)
    df = df.dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    print(f"  Training set: {len(X):,} samples")
    print("  Label distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        print(f"    {label}: {count} ({count / len(y):.1%})")
    print()

    tscv = TimeSeriesSplit(n_splits=4)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        mdl = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight="balanced", random_state=42,
        )
        mdl.fit(X[tr_idx], y[tr_idx])
        preds = mdl.predict(X[val_idx])
        print(f"Fold {fold + 1}:")
        print(classification_report(y[val_idx], preds))

    final = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        class_weight="balanced", random_state=42,
    )
    final.fit(X, y)

    # Feature importance
    importances = sorted(zip(FEATURE_COLS, final.feature_importances_), key=lambda x: -x[1])
    print("Top features:")
    for name, imp in importances[:6]:
        print(f"  {name}: {imp:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_cols": FEATURE_COLS}, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
