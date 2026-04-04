#!/usr/bin/env python3
"""F: Fit Isolation Forest anomaly detector (B4).

Detects abnormal market microstructure (OI spikes, volume anomalies).
Refit weekly on recent 15m data.

Usage:
    python scripts/train_anomaly.py
    python scripts/train_anomaly.py --rows 10000 --contamination 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


FEATURE_COLS = ["vol_ratio", "price_vel", "high_low_rng"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe",     default="5m", choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Candle timeframe; auto-selects --csv and --out if not overridden.")
    parser.add_argument("--symbol",        default=None, help="Symbol e.g. btcusdt — prefixes output filename")
    parser.add_argument("--csv",           default=None)
    parser.add_argument("--out",           default=None)
    parser.add_argument("--rows",          type=int,   default=10_000)
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="Expected fraction of anomalies. Start conservative (0.01).")
    args = parser.parse_args()

    sym = args.symbol.lower() if args.symbol else "btcusdt"
    sym_prefix = f"{sym}_" if args.symbol else ""
    if args.csv is None:
        args.csv = f"data/ohlcv/{sym}_{args.timeframe}.csv"
    if args.out is None:
        args.out = f"models/isolation_forest_{sym_prefix}{args.timeframe}.joblib"

    print(f"Loading {args.csv} (last {args.rows:,} rows) …")
    df = pd.read_csv(args.csv)
    df = df.sort_values("timestamp").tail(args.rows).reset_index(drop=True)
    print(f"  {len(df):,} rows loaded")

    c = df["close"]
    v = df["volume"]
    h = df["high"]
    l = df["low"]

    df["vol_ratio"]    = v / v.rolling(20).mean()
    df["price_vel"]    = c.pct_change(3) * 100          # 3-candle price velocity %
    df["high_low_rng"] = (h - l) / ((h + l) / 2) * 100  # candle body range %

    df = df.dropna(subset=FEATURE_COLS)
    X  = df[FEATURE_COLS].values

    print(f"  Training on {len(X):,} samples  contamination={args.contamination}")

    model = IsolationForest(
        n_estimators=200,
        contamination=args.contamination,
        random_state=42,
    )
    model.fit(X)

    anomaly_count = (model.predict(X) == -1).sum()
    print(f"  Anomalies in training set: {anomaly_count} ({anomaly_count/len(X):.1%})")

    # Show feature stats for context
    print("\nFeature statistics:")
    for col in FEATURE_COLS:
        arr = df[col].values
        print(f"  {col:>14}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"p95={np.percentile(arr, 95):.3f}  p99={np.percentile(arr, 99):.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
