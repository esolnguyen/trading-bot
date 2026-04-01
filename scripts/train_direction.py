#!/usr/bin/env python3
"""F: Train XGBoost direction classifier (B2).

Trains on 15m OHLCV features. Retrain weekly after ~500 new candles.

Usage:
    python scripts/train_direction.py
    python scripts/train_direction.py --csv data/ethusdt_15m.csv --out models/xgboost_direction_eth.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator features matching live IndicatorCalculator output."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_line"]   = macd
    df["macd_signal"] = sig
    df["macd_hist"]   = macd - sig

    # EMA 20 / 50
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_spread"] = df["ema_20"] - df["ema_50"]

    # ATR-14
    prev_c  = c.shift(1)
    tr      = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    # ADX-14 (simplified DI-based)
    up   = h.diff()
    down = -l.diff()
    pdm  = up.where((up > down) & (up > 0), 0.0)
    mdm  = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    pdi  = 100 * pdm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    mdi  = 100 * mdm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    # Bollinger Bands (20, 2σ)
    mid  = c.rolling(20).mean()
    std  = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - 2 * std
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pos"]   = (c - df["bb_lower"]) / bb_width

    # OBV slope (20-period linear regression slope, normalised)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(20) / obv.abs().rolling(20).mean().replace(0, np.nan)

    # Volume ratio
    df["vol_ratio"] = v / v.rolling(20).mean()

    # Choppiness index (14)
    atr_sum = tr.rolling(14).sum()
    hh14    = h.rolling(14).max()
    ll14    = l.rolling(14).min()
    rng14   = (hh14 - ll14).replace(0, np.nan)
    df["choppiness"] = 100 * np.log10(atr_sum / rng14) / np.log10(14)

    # Lag features t-1, t-2, t-3
    for col in ["rsi_14", "macd_hist", "adx"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "ema_spread", "atr_pct", "adx", "bb_pos", "obv_slope",
    "vol_ratio", "choppiness",
    "rsi_14_lag1", "rsi_14_lag2", "rsi_14_lag3",
    "macd_hist_lag1", "macd_hist_lag2", "macd_hist_lag3",
    "adx_lag1", "adx_lag2", "adx_lag3",
]

LOOKAHEAD = 6   # candles — 30 min ahead on 5m, 8 min on 1m, 2h on 15m, 8h on 1h
THRESHOLD = 0.002  # 0.2% move to call it bullish (at X50 leverage = 10% gain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="5m", choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Candle timeframe; auto-selects --csv and --out if not overridden.")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.csv is None:
        args.csv = f"data/ohlcv/btcusdt_{args.timeframe}.csv"
    if args.out is None:
        args.out = f"models/xgboost_direction_{args.timeframe}.joblib"

    print(f"Loading {args.csv} …")
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded")

    df = compute_features(df)

    # Target: did price rise > THRESHOLD within LOOKAHEAD candles?
    # CRITICAL: use shift(-LOOKAHEAD) — never include future data in features
    df["target"] = (df["close"].shift(-LOOKAHEAD) / df["close"] - 1 > THRESHOLD).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["target"])
    # Drop the last LOOKAHEAD rows (no valid future target)
    df = df.iloc[:-LOOKAHEAD]

    X = df[FEATURE_COLS].values
    y = df["target"].values
    bullish_ratio = y.mean()
    # Weight the minority class inversely to its frequency so the model doesn't
    # just predict "bearish" every time (critical for short timeframes like 5m/1m).
    scale_pos_weight = (1 - bullish_ratio) / bullish_ratio if bullish_ratio > 0 else 1.0
    print(f"  Training set: {len(X):,} samples  class balance: {bullish_ratio:.1%} bullish  scale_pos_weight={scale_pos_weight:.2f}\n")

    # Time-series cross-validation — NEVER shuffle financial data
    tscv = TimeSeriesSplit(n_splits=5)
    aucs: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        mdl = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", verbosity=0,
        )
        mdl.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds_proba = mdl.predict_proba(X[val_idx])[:, 1]
        preds       = (preds_proba >= 0.5).astype(int)
        auc         = roc_auc_score(y[val_idx], preds_proba)
        aucs.append(auc)
        print(f"Fold {fold + 1}  AUC={auc:.3f}")
        print(classification_report(y[val_idx], preds, target_names=["bearish", "bullish"]))

    print(f"Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")

    # Final model on all data
    final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", verbosity=0,
    )
    final.fit(X, y)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_cols": FEATURE_COLS}, args.out)
    print(f"Saved → {args.out}  ({len(X):,} training samples)")


if __name__ == "__main__":
    main()
