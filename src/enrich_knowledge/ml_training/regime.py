"""Driver: Random Forest macro regime classifier.

Classifies BULL_TRENDING / BULL_CORRECTION / BEAR_TRENDING /
ACCUMULATION from daily-equivalent features. Labels are rule-based;
the RF generalises beyond the rules via TimeSeriesSplit CV.

Usage:
    python -m src.enrich_knowledge.runners.run_training --model regime
    python -m src.enrich_knowledge.ml_training.regime --symbol btcusdt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

from src.enrich_knowledge.config import EnrichKnowledgeSettings

logger = logging.getLogger(__name__)

_REGIME_TIMEFRAME = "1d"

_CPD = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}

FEATURE_COLS = [
    "ema50_dist", "ema100_dist", "ema200_dist",
    "ema50_slope", "ema200_slope",
    "high_52w_dist", "low_52w_dist",
    "adx_14", "realized_vol",
    "hh_count", "ll_count",
]


def compute_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    cpd = _CPD[timeframe]

    for span in [50, 100, 200]:
        ema = c.ewm(span=span * cpd, adjust=False).mean()
        df[f"ema{span}_dist"] = (c - ema) / ema
        df[f"ema{span}_slope"] = ema.diff(5 * cpd) / ema

    w365 = 365 * cpd
    w90 = 90 * cpd
    w30 = 30 * cpd
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

    df["realized_vol"] = c.pct_change().rolling(w30).std() * np.sqrt(365 * cpd)

    df["hh_count"] = h.rolling(w90).apply(
        lambda x: sum(x[i] > x[i - 1] for i in range(1, len(x))), raw=True
    )
    df["ll_count"] = l.rolling(w90).apply(
        lambda x: sum(x[i] < x[i - 1] for i in range(1, len(x))), raw=True
    )

    return df


def label_regime(row: pd.Series) -> str:
    e200 = row["ema200_dist"]
    sl50 = row["ema50_slope"]
    h52 = row["high_52w_dist"]

    if e200 > 0.05 and sl50 > 0:
        return "BULL_TRENDING"
    if e200 > -0.05 and h52 < -0.10:
        return "BULL_CORRECTION"
    if e200 < -0.05 and sl50 < 0:
        return "BEAR_TRENDING"
    return "ACCUMULATION"


def fit(
    *,
    symbol: str,
    timeframe: str = _REGIME_TIMEFRAME,
    csv: str | None = None,
    out: str | None = None,
) -> None:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_{timeframe}.csv"
    out_path = out or f"models/{timeframe}/regime_classifier_{sym}.joblib"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows loaded  [{timeframe}]")

    df = compute_features(df, timeframe=timeframe)
    df["label"] = df.apply(label_regime, axis=1)
    df = df.dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    print(f"  Training set: {len(X):,} samples")
    print("  Label distribution:")
    for label, count in zip(*np.unique(y, return_counts=True), strict=False):
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

    importances = sorted(
        zip(FEATURE_COLS, final.feature_importances_, strict=False),
        key=lambda x: -x[1],
    )
    print("Top features:")
    for name, imp in importances[:6]:
        print(f"  {name}: {imp:.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_cols": FEATURE_COLS}, out_path)
    print(f"\nSaved -> {out_path}")


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for symbol in settings.ml_training.training_symbols:
        if dry_run:
            logger.info("[dry-run] would fit regime %s %s", symbol, _REGIME_TIMEFRAME)
            continue
        try:
            fit(symbol=symbol, timeframe=_REGIME_TIMEFRAME)
        except Exception:
            logger.exception("regime fit failed for %s", symbol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe",
        default=_REGIME_TIMEFRAME,
        choices=list(_CPD.keys()),
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
