"""Walk-forward backtest for the IsolationForest anomaly detector.

Refits the detector every ``refit_every_days`` on a rolling
``lookback_days`` window, scores the following ``refit_every_days`` of
candles, and measures predictive validity: when a candle is flagged as
anomalous, does a "big move" actually follow within ``horizon`` candles?

A big move = max forward run-up OR max forward drawdown over ``horizon``
candles exceeds ``abs_threshold`` (default 2%). The metric grid:

  * Flag rate    — fraction of candles flagged. Calibration check.
  * Base rate    — fraction of candles followed by a big move. Reference.
  * Precision    — P(big move | flagged). Higher = fewer false alarms.
  * Recall       — P(flagged | big move). Higher = fewer misses.
  * Lift         — precision / base_rate. > 1 means the flag carries
                   predictive information beyond chance.

Lift is the single number that answers "is this gate worth keeping?"

Usage:
    python -m src.enrich_knowledge.ml_training.backtest_anomaly --symbol btcusdt --timeframe 1h
    python -m src.enrich_knowledge.ml_training.backtest_anomaly --symbol btcusdt --timeframe 4h --horizon 6 --abs-threshold 0.02
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .anomaly import FEATURE_COLS

logger = logging.getLogger(__name__)

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}


def _build_model(contamination: float) -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )


def backtest(
    *,
    symbol: str,
    timeframe: str,
    lookback_days: int = 180,
    refit_every_days: int = 7,
    horizon: int = 4,
    abs_threshold: float = 0.02,
    contamination: float = 0.01,
    csv: str | None = None,
    out: str | None = None,
) -> dict[str, float]:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_{timeframe}.csv"
    out_path = out or f"models/{timeframe}/backtest_anomaly_{sym}.csv"
    cpd = _CANDLES_PER_DAY[timeframe]
    train_window = lookback_days * cpd
    refit_step = refit_every_days * cpd

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded  ({timeframe})")

    c = df["close"]
    v = df["volume"]
    h = df["high"]
    low = df["low"]
    df["vol_ratio"] = v / v.rolling(20).mean()
    df["price_vel"] = c.pct_change(3) * 100
    df["high_low_rng"] = (h - low) / ((h + low) / 2) * 100

    # Forward big-move label (peak run-up or drawdown within horizon).
    # shift(-horizon) means we're looking at future bars relative to the
    # current candle, then we compare against close[t].
    fwd_high = h.rolling(horizon).max().shift(-horizon)
    fwd_low = low.rolling(horizon).min().shift(-horizon)
    df["fwd_runup"] = (fwd_high - c) / c
    df["fwd_drawdown"] = (c - fwd_low) / c
    df["fwd_big_move"] = (
        (df["fwd_runup"] > abs_threshold) | (df["fwd_drawdown"] > abs_threshold)
    ).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["fwd_big_move"]).reset_index(drop=True)
    df = df.iloc[:-horizon].reset_index(drop=True)

    if len(df) < train_window + refit_step:
        raise SystemExit(
            f"Not enough data: {len(df):,} rows < train_window {train_window:,} "
            f"+ refit_step {refit_step:,}. Backfill more candles."
        )

    n_steps = (len(df) - train_window) // refit_step
    print(
        f"  Walk-forward: {n_steps} folds, refit every {refit_every_days}d "
        f"on rolling {lookback_days}d window  [ANOMALY]"
    )
    print(
        f"  Big move = |fwd run-up| > {abs_threshold:.1%} OR "
        f"|fwd drawdown| > {abs_threshold:.1%} within {horizon} candles\n"
    )

    flag_chunks: list[np.ndarray] = []
    big_chunks: list[np.ndarray] = []
    runup_chunks: list[np.ndarray] = []
    dd_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []

    for i in range(n_steps):
        train_start = i * refit_step
        train_end = train_start + train_window
        test_end = min(train_end + refit_step, len(df))

        # Embargo last ``horizon`` rows of train: their fwd_big_move
        # label depends on candles inside the test window.
        train = df.iloc[train_start : train_end - horizon]
        test = df.iloc[train_end:test_end]
        if len(train) == 0 or len(test) == 0:
            continue

        X_tr = train[FEATURE_COLS].values
        X_te = test[FEATURE_COLS].values
        model = _build_model(contamination)
        model.fit(X_tr)
        # IsolationForest.predict: -1 = anomaly, +1 = normal
        flags = (model.predict(X_te) == -1).astype(int)

        flag_chunks.append(flags)
        big_chunks.append(test["fwd_big_move"].values)
        runup_chunks.append(test["fwd_runup"].values)
        dd_chunks.append(test["fwd_drawdown"].values)
        ts_chunks.append(test["timestamp"].values)

        if (i + 1) % 20 == 0 or i == n_steps - 1:
            d0 = pd.Timestamp(train.iloc[0]["timestamp"]).date()
            d1 = pd.Timestamp(train.iloc[-1]["timestamp"]).date()
            fr = flags.mean()
            print(f"  fold {i + 1:>3}/{n_steps}  train [{d0}..{d1}]  flag rate={fr:.3%}")

    flags_all = np.concatenate(flag_chunks)
    big_all = np.concatenate(big_chunks)
    runup_all = np.concatenate(runup_chunks)
    dd_all = np.concatenate(dd_chunks)
    ts_all = np.concatenate(ts_chunks)

    base_rate = big_all.mean()
    flag_rate = flags_all.mean()
    n_flagged = int(flags_all.sum())
    n_big = int(big_all.sum())

    precision = big_all[flags_all == 1].mean() if n_flagged else float("nan")
    recall = flags_all[big_all == 1].mean() if n_big else float("nan")
    lift = precision / base_rate if base_rate > 0 else float("nan")

    # Magnitude conditioning: when flagged, how big is the move on average?
    avg_runup_flagged = runup_all[flags_all == 1].mean() if n_flagged else float("nan")
    avg_runup_clean = runup_all[flags_all == 0].mean()
    avg_dd_flagged = dd_all[flags_all == 1].mean() if n_flagged else float("nan")
    avg_dd_clean = dd_all[flags_all == 0].mean()

    print("\n=== Walk-forward summary [ANOMALY] ===")
    print(f"  Out-of-sample candles:    {len(flags_all):,}")
    print(f"  Flag rate:                {flag_rate:.4f}  ({n_flagged:,} flagged)")
    print(f"  Base rate (big move):     {base_rate:.4f}  ({n_big:,} big moves)")
    print(f"  Precision P(big|flag):    {precision:.4f}")
    print(f"  Recall    P(flag|big):    {recall:.4f}")
    verdict = "predictive" if lift > 1.1 else "no edge" if lift < 1.05 else "marginal"
    print(f"  Lift (precision/base):    {lift:.3f}    [{verdict}]")
    print()
    print("  Avg fwd run-up:")
    print(f"    flagged:                {avg_runup_flagged:.4f}")
    print(f"    clean:                  {avg_runup_clean:.4f}")
    print("  Avg fwd drawdown:")
    print(f"    flagged:                {avg_dd_flagged:.4f}")
    print(f"    clean:                  {avg_dd_clean:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": ts_all,
        "flag": flags_all,
        "fwd_runup": runup_all,
        "fwd_drawdown": dd_all,
        "fwd_big_move": big_all,
    }).to_csv(out_path, index=False)
    print(f"\nSaved per-prediction trace -> {out_path}")

    return {
        "flag_rate": float(flag_rate),
        "base_rate": float(base_rate),
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(lift),
        "n_predictions": float(len(flags_all)),
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
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--abs-threshold", type=float, default=0.02)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        refit_every_days=args.refit_every_days,
        horizon=args.horizon,
        abs_threshold=args.abs_threshold,
        contamination=args.contamination,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
