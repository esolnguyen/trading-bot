#!/usr/bin/env python3
"""F: Train LogReg trade outcome predictor (B3).

Uses closed trade history from data/trade_history_*.json.
Retrain after every 50 new closed trades.

Usage:
    python scripts/train_outcome.py
    python scripts/train_outcome.py --min-trades 30
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


FEATURE_COLS = ["rsi", "adx", "atr_pct", "trend", "vol_state", "bb_pos"]


def load_trades(data_dir: str = "data") -> pd.DataFrame:
    rows = []
    for path in glob.glob(f"{data_dir}/trade_history_*.json"):
        with open(path) as f:
            trades = json.load(f)
        if not isinstance(trades, list):
            continue
        for t in trades:
            action = str(t.get("action", ""))
            if not action.startswith("CLOSE"):
                continue
            cond = t.get("market_conditions", {}) or {}
            pnl  = t.get("pnl_pct") or t.get("pnl") or 0.0
            rows.append({
                "rsi":       float(cond.get("rsi", 50)),
                "adx":       float(cond.get("adx", 20)),
                "atr_pct":   float(cond.get("atr_percentage", 1.5)),
                "trend":     1.0  if cond.get("trend_direction") == "BULLISH" else
                            (-1.0 if cond.get("trend_direction") == "BEARISH" else 0.0),
                "vol_state": 1.0  if cond.get("volume_state") == "HIGH" else 0.0,
                "bb_pos":    {"UPPER": 1.0, "MIDDLE": 0.0, "LOWER": -1.0}.get(
                                 str(cond.get("bb_position", "MIDDLE")), 0.0),
                "profitable": 1 if float(pnl) > 0 else 0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--out",        default="models/outcome_predictor.joblib")
    parser.add_argument("--min-trades", type=int, default=50)
    args = parser.parse_args()

    print("Loading trade history …")
    df = load_trades(args.data_dir)
    print(f"  Found {len(df)} closed trades")

    if len(df) < args.min_trades:
        print(f"  Only {len(df)} trades — need at least {args.min_trades}. Run again later.")
        sys.exit(0)

    X = df[FEATURE_COLS].values
    y = df["profitable"].values
    print(f"  Win rate: {y.mean():.1%}  ({y.sum()} wins / {len(y)} trades)\n")

    # Stratified k-fold (trade data is small — can't use time-series split)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")),
    ])
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring="roc_auc")
    print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)
    preds = model.predict(X)
    print("\nFull training set report:")
    print(classification_report(y, preds, target_names=["loss", "win"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, args.out)
    print(f"Saved → {args.out}  ({len(X)} trades)")


if __name__ == "__main__":
    main()
