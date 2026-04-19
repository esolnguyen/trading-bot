"""Driver: XGBoost direction classifier.

Fits one model per (symbol, timeframe). Target: did price rise by more
than THRESHOLD within LOOKAHEAD candles? Uses TimeSeriesSplit — never
shuffle financial data.

Usage:
    python -m src.enrich_knowledge.runners.run_training --model direction
    python -m src.enrich_knowledge.ml_training.direction --symbol btcusdt --timeframe 4h
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.enrich_knowledge.config import EnrichKnowledgeSettings

logger = logging.getLogger(__name__)

LOOKAHEAD = 6
THRESHOLD = 0.002

FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "ema_spread", "ema_20_dist", "ema_50_dist",
    "atr_pct", "adx", "bb_pos", "obv_slope",
    "vol_ratio", "choppiness", "cci_14",
    "rsi_14_lag1", "rsi_14_lag2", "rsi_14_lag3",
    "macd_hist_lag1", "macd_hist_lag2", "macd_hist_lag3",
    "adx_lag1", "adx_lag2", "adx_lag3",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicator features matching live IndicatorCalculator output."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = macd - sig

    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_spread"] = df["ema_20"] - df["ema_50"]
    df["ema_20_dist"] = (c - df["ema_20"]) / df["ema_20"].replace(0, np.nan) * 100
    df["ema_50_dist"] = (c - df["ema_50"]) / df["ema_50"].replace(0, np.nan) * 100

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    up = h.diff()
    down = -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    pdi = 100 * pdm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    mdi = 100 * mdm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_mid"] = mid
    df["bb_lower"] = mid - 2 * std
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pos"] = (c - df["bb_lower"]) / bb_width

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(20) / obv.abs().rolling(20).mean().replace(0, np.nan)

    df["vol_ratio"] = v / v.rolling(20).mean()

    atr_sum = tr.rolling(14).sum()
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    rng14 = (hh14 - ll14).replace(0, np.nan)
    df["choppiness"] = 100 * np.log10(atr_sum / rng14) / np.log10(14)

    tp = (h + l + c) / 3.0
    tp_mean = tp.rolling(14).mean()
    tp_mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_14"] = (tp - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

    for col in ["rsi_14", "macd_hist", "adx"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def fit(
    *,
    symbol: str,
    timeframe: str,
    csv: str | None = None,
    out: str | None = None,
) -> None:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_{timeframe}.csv"
    out_path = out or f"models/{timeframe}/xgboost_direction_{sym}.joblib"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded")

    df = compute_features(df)

    # shift(-LOOKAHEAD) — never include future data in features
    df["target"] = (df["close"].shift(-LOOKAHEAD) / df["close"] - 1 > THRESHOLD).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["target"])
    df = df.iloc[:-LOOKAHEAD]

    X = df[FEATURE_COLS].values
    y = df["target"].values
    bullish_ratio = y.mean()
    scale_pos_weight = (1 - bullish_ratio) / bullish_ratio if bullish_ratio > 0 else 1.0
    print(
        f"  Training set: {len(X):,} samples  class balance: {bullish_ratio:.1%} bullish  "
        f"scale_pos_weight={scale_pos_weight:.2f}\n"
    )

    tscv = TimeSeriesSplit(n_splits=5)
    aucs: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        mdl = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", verbosity=0,
        )
        mdl.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        preds_proba = mdl.predict_proba(X[val_idx])[:, 1]
        preds = (preds_proba >= 0.5).astype(int)
        auc = roc_auc_score(y[val_idx], preds_proba)
        aucs.append(auc)
        print(f"Fold {fold + 1}  AUC={auc:.3f}")
        print(classification_report(y[val_idx], preds, target_names=["bearish", "bullish"]))

    print(f"Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")

    final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", verbosity=0,
    )
    final.fit(X, y)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_cols": FEATURE_COLS}, out_path)
    print(f"Saved → {out_path}  ({len(X):,} training samples)")


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for timeframe in settings.ml_training.ml_timeframes:
        for symbol in settings.ml_training.training_symbols:
            if dry_run:
                logger.info("[dry-run] would fit direction %s %s", symbol, timeframe)
                continue
            try:
                fit(symbol=symbol, timeframe=timeframe)
            except Exception:
                logger.exception("direction fit failed for %s %s", symbol, timeframe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe",
        default="5m",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    fit(
        symbol=args.symbol,
        timeframe=args.timeframe,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
