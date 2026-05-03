"""Walk-forward backtest for the XGBoost direction classifier.

Sweeps a rolling 180-day window across the entire OHLCV history,
refitting every N days and scoring the next N days out-of-sample.
Reports pooled AUC / accuracy / Brier on the cumulative held-out
predictions — the honest measure of model edge under live conditions.

Unlike the in-sample TimeSeriesSplit in ``direction.fit``, this never
trains and tests on the same regime: every test window comes from
candles the model has never seen, and the rolling lookback simulates
the production retrain cadence.

Usage:
    python -m src.enrich_knowledge.ml_training.backtest_direction \\
        --symbol btcusdt --timeframe 1h
    python -m src.enrich_knowledge.ml_training.backtest_direction \\
        --symbol btcusdt --timeframe 4h --refit-every-days 7 --lookback-days 180
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from .direction import FEATURE_COLS, LOOKAHEAD, THRESHOLD, compute_features

logger = logging.getLogger(__name__)

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}


def _build_model(scale_pos_weight: float) -> XGBClassifier:
    # n_jobs=1: per-fold training data (~1k rows) is too small for
    # XGBoost's threading to amortise — measured 250x speedup vs the
    # multi-threaded default on this workload.
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", verbosity=0, n_jobs=1,
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
    out_path = out or f"models/{timeframe}/backtest_direction_{sym}.csv"
    cpd = _CANDLES_PER_DAY[timeframe]
    train_window = lookback_days * cpd
    refit_step = refit_every_days * cpd

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded  ({timeframe})")

    df = compute_features(df)
    df["target"] = (df["close"].shift(-LOOKAHEAD) / df["close"] - 1 > THRESHOLD).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    df = df.iloc[:-LOOKAHEAD].reset_index(drop=True)

    if len(df) < train_window + refit_step:
        raise SystemExit(
            f"Not enough data: {len(df):,} rows < train_window {train_window:,} "
            f"+ refit_step {refit_step:,}. Backfill more candles."
        )

    n_steps = (len(df) - train_window) // refit_step
    print(
        f"  Walk-forward: {n_steps} folds, refit every {refit_every_days}d "
        f"on rolling {lookback_days}d window\n"
    )

    proba_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    fold_aucs: list[float] = []

    for i in range(n_steps):
        train_start = i * refit_step
        train_end = train_start + train_window
        test_end = min(train_end + refit_step, len(df))

        # Embargo the last LOOKAHEAD rows of train: their targets depend
        # on close prices that fall inside the test window, so leaving
        # them in would leak future information across the boundary.
        train = df.iloc[train_start : train_end - LOOKAHEAD]
        test = df.iloc[train_end:test_end]
        if len(train) == 0 or len(test) == 0:
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train["target"].values
        X_te = test[FEATURE_COLS].values
        y_te = test["target"].values

        bull_ratio = y_tr.mean()
        spw = (1 - bull_ratio) / bull_ratio if 0 < bull_ratio < 1 else 1.0
        model = _build_model(spw)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        proba_chunks.append(proba)
        actual_chunks.append(y_te)
        ts_chunks.append(test["timestamp"].values)

        fold_auc = (
            roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")
        )
        fold_aucs.append(fold_auc)

        if (i + 1) % 10 == 0 or i == n_steps - 1:
            d0 = pd.Timestamp(train.iloc[0]["timestamp"]).date()
            d1 = pd.Timestamp(train.iloc[-1]["timestamp"]).date()
            print(f"  fold {i + 1:>3}/{n_steps}  train [{d0}..{d1}]  fold AUC={fold_auc:.3f}")

    proba_all = np.concatenate(proba_chunks)
    actual_all = np.concatenate(actual_chunks)
    ts_all = np.concatenate(ts_chunks)
    pred_all = (proba_all >= 0.5).astype(int)

    auc = roc_auc_score(actual_all, proba_all)
    acc = (pred_all == actual_all).mean()
    brier = brier_score_loss(actual_all, proba_all)
    base_rate = actual_all.mean()
    hit_when_bull = actual_all[pred_all == 1].mean() if pred_all.sum() else float("nan")
    fold_arr = np.array(fold_aucs)

    print("\n=== Walk-forward summary ===")
    print(f"  Out-of-sample predictions: {len(actual_all):,}")
    print(f"  Pooled AUC:         {auc:.4f}")
    print(
        f"  Per-fold AUC:       mean={np.nanmean(fold_arr):.3f}  "
        f"std={np.nanstd(fold_arr):.3f}  "
        f"min={np.nanmin(fold_arr):.3f}  max={np.nanmax(fold_arr):.3f}"
    )
    print(f"  Accuracy @0.5:      {acc:.4f}")
    print(f"  Brier score:        {brier:.4f}  (lower=better)")
    print(f"  Base rate (bull%):  {base_rate:.4f}")
    print(f"  Hit when pred=bull: {hit_when_bull:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": ts_all,
        "proba": proba_all,
        "actual": actual_all,
        "pred": pred_all,
    }).to_csv(out_path, index=False)
    print(f"\nSaved per-prediction trace -> {out_path}")

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "brier": float(brier),
        "base_rate": float(base_rate),
        "n_predictions": float(len(actual_all)),
        "n_folds": float(n_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument(
        "--timeframe", default="1h", choices=list(_CANDLES_PER_DAY.keys())
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
