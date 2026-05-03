"""Walk-forward backtest — regression variant of backtest_direction.

Same feature set, same window, same refit cadence as the classifier
backtest. Difference: target is the raw 6-bar return (continuous), and
the model is XGBRegressor with MSE loss.

For evaluation we project the regressor's continuous output back onto
the classifier's metric grid:

  * AUC: predicted_return is used directly as a ranking score against
    the binary target ``ret > THRESHOLD``. Apples-to-apples vs the
    classifier's predicted probability.
  * Accuracy / hit rate: ``predicted_return > THRESHOLD`` -> bullish.
  * Plus regression-native metrics (MAE, R^2) so the regressor's own
    objective is also visible.

Usage:
    python -m src.enrich_knowledge.ml_training.backtest_direction_regression \\
        --symbol btcusdt --timeframe 4h
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

from .direction import FEATURE_COLS, LOOKAHEAD, THRESHOLD, compute_features

logger = logging.getLogger(__name__)

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}


def _build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", verbosity=0, n_jobs=1,
    )


def backtest(
    *,
    symbol: str,
    timeframe: str,
    lookback_days: int = 180,
    refit_every_days: int = 7,
    csv: str | None = None,
    out: str | None = None,
) -> dict[str, float]:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_{timeframe}.csv"
    out_path = out or f"models/{timeframe}/backtest_direction_regression_{sym}.csv"
    cpd = _CANDLES_PER_DAY[timeframe]
    train_window = lookback_days * cpd
    refit_step = refit_every_days * cpd

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded  ({timeframe})")

    df = compute_features(df)
    df["ret"] = df["close"].shift(-LOOKAHEAD) / df["close"] - 1
    df["target"] = (df["ret"] > THRESHOLD).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ["ret", "target"]).reset_index(drop=True)
    df = df.iloc[:-LOOKAHEAD].reset_index(drop=True)

    if len(df) < train_window + refit_step:
        raise SystemExit(
            f"Not enough data: {len(df):,} rows < train_window {train_window:,} "
            f"+ refit_step {refit_step:,}. Backfill more candles."
        )

    n_steps = (len(df) - train_window) // refit_step
    print(
        f"  Walk-forward: {n_steps} folds, refit every {refit_every_days}d "
        f"on rolling {lookback_days}d window  [REGRESSION]\n"
    )

    pred_chunks: list[np.ndarray] = []
    actual_ret_chunks: list[np.ndarray] = []
    actual_bin_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    fold_aucs: list[float] = []

    for i in range(n_steps):
        train_start = i * refit_step
        train_end = train_start + train_window
        test_end = min(train_end + refit_step, len(df))

        train = df.iloc[train_start : train_end - LOOKAHEAD]
        test = df.iloc[train_end:test_end]
        if len(train) == 0 or len(test) == 0:
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train["ret"].values
        X_te = test[FEATURE_COLS].values
        y_te_ret = test["ret"].values
        y_te_bin = test["target"].values

        model = _build_model()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        pred_chunks.append(pred)
        actual_ret_chunks.append(y_te_ret)
        actual_bin_chunks.append(y_te_bin)
        ts_chunks.append(test["timestamp"].values)

        fold_auc = (
            roc_auc_score(y_te_bin, pred) if len(np.unique(y_te_bin)) > 1 else float("nan")
        )
        fold_aucs.append(fold_auc)

        if (i + 1) % 10 == 0 or i == n_steps - 1:
            d0 = pd.Timestamp(train.iloc[0]["timestamp"]).date()
            d1 = pd.Timestamp(train.iloc[-1]["timestamp"]).date()
            print(f"  fold {i + 1:>3}/{n_steps}  train [{d0}..{d1}]  fold AUC={fold_auc:.3f}")

    pred_all = np.concatenate(pred_chunks)
    ret_all = np.concatenate(actual_ret_chunks)
    bin_all = np.concatenate(actual_bin_chunks)
    ts_all = np.concatenate(ts_chunks)
    pred_bin = (pred_all > THRESHOLD).astype(int)

    auc = roc_auc_score(bin_all, pred_all)
    acc = (pred_bin == bin_all).mean()
    base_rate = bin_all.mean()
    hit_when_bull = bin_all[pred_bin == 1].mean() if pred_bin.sum() else float("nan")
    mae = mean_absolute_error(ret_all, pred_all)
    r2 = r2_score(ret_all, pred_all)
    fold_arr = np.array(fold_aucs)

    print("\n=== Walk-forward summary [REGRESSION] ===")
    print(f"  Out-of-sample predictions: {len(ret_all):,}")
    print(f"  Pooled AUC (rank score):  {auc:.4f}")
    print(
        f"  Per-fold AUC:             mean={np.nanmean(fold_arr):.3f}  "
        f"std={np.nanstd(fold_arr):.3f}  "
        f"min={np.nanmin(fold_arr):.3f}  max={np.nanmax(fold_arr):.3f}"
    )
    print(f"  Accuracy @THRESHOLD:      {acc:.4f}")
    print(f"  Base rate (bull%):        {base_rate:.4f}")
    print(f"  Hit when pred=bull:       {hit_when_bull:.4f}")
    print(f"  MAE (ret):                {mae:.5f}")
    print(f"  R^2 (ret):                {r2:.5f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": ts_all,
        "pred_ret": pred_all,
        "actual_ret": ret_all,
        "actual_bin": bin_all,
        "pred_bin": pred_bin,
    }).to_csv(out_path, index=False)
    print(f"\nSaved per-prediction trace -> {out_path}")

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "mae": float(mae),
        "r2": float(r2),
        "base_rate": float(base_rate),
        "n_predictions": float(len(ret_all)),
        "n_folds": float(n_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument(
        "--timeframe", default="4h", choices=list(_CANDLES_PER_DAY.keys())
    )
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--refit-every-days", type=int, default=7)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        refit_every_days=args.refit_every_days,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
