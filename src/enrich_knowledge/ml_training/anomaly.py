"""Driver: Isolation Forest anomaly detector.

Fits one model per (symbol, timeframe) from cached OHLCV CSVs.
Features encode microstructure (volume ratio, price velocity, candle
range) so the detector flags OI/volume spikes at inference time.

Usage:
    # From the batch runner (every configured symbol x timeframe):
    python -m src.enrich_knowledge.runners.run_training --model anomaly

    # Ad-hoc single fit:
    python -m src.enrich_knowledge.ml_training.anomaly --symbol btcusdt --timeframe 4h
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.enrich_knowledge.config import EnrichKnowledgeSettings

logger = logging.getLogger(__name__)

FEATURE_COLS = ["vol_ratio", "price_vel", "high_low_rng"]

# Microstructure baselines (volume patterns, velocity regimes) drift
# with venue/fee/HFT changes — training on years of candles teaches the
# detector "normal" behaviour that no longer applies. 180d ≈ 6 months.
DEFAULT_LOOKBACK_DAYS = 180

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}


def fit(
    *,
    symbol: str,
    timeframe: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    rows: int | None = None,
    contamination: float = 0.01,
    csv: str | None = None,
    out: str | None = None,
) -> None:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_{timeframe}.csv"
    out_path = out or f"models/{timeframe}/isolation_forest_{sym}.joblib"

    if rows is None:
        rows = lookback_days * _CANDLES_PER_DAY.get(timeframe, 96)

    print(f"Loading {csv_path} (last {rows:,} rows, ~{lookback_days}d on {timeframe}) …")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").tail(rows).reset_index(drop=True)
    print(f"  {len(df):,} rows loaded")

    c = df["close"]
    v = df["volume"]
    h = df["high"]
    l = df["low"]

    df["vol_ratio"] = v / v.rolling(20).mean()
    df["price_vel"] = c.pct_change(3) * 100
    df["high_low_rng"] = (h - l) / ((h + l) / 2) * 100

    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values

    print(f"  Training on {len(X):,} samples  contamination={contamination}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)

    anomaly_count = (model.predict(X) == -1).sum()
    print(f"  Anomalies in training set: {anomaly_count} ({anomaly_count/len(X):.1%})")

    print("\nFeature statistics:")
    for col in FEATURE_COLS:
        arr = df[col].values
        print(
            f"  {col:>14}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
            f"p95={np.percentile(arr, 95):.3f}  p99={np.percentile(arr, 99):.3f}"
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, out_path)
    print(f"\nSaved → {out_path}")


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for timeframe in settings.ml_training.ml_timeframes:
        for symbol in settings.ml_training.training_symbols:
            if dry_run:
                logger.info("[dry-run] would fit anomaly %s %s", symbol, timeframe)
                continue
            try:
                fit(symbol=symbol, timeframe=timeframe)
            except Exception:
                logger.exception("anomaly fit failed for %s %s", symbol, timeframe)


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
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Cap training data to the most recent N days of candles.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Raw row cap — overrides --lookback-days when set.",
    )
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    fit(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        rows=args.rows,
        contamination=args.contamination,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
