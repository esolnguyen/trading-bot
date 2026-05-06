"""Driver: Random Forest macro regime classifier (forward-realized labels).

The previous label rule was a hand-coded threshold over EMA200-distance,
EMA50-slope, and 52w-high-distance — features that were ALSO in the
training set. The Random Forest had no signal to learn beyond memorizing
the rule. This driver replaces the rule labels with FORWARD-realized
outcomes: walk forward FORWARD_HORIZON_DAYS calendar days from each bar
and assign the regime based on the actual return + drawdown + runup that
followed. The model now generalizes on real market structure rather than
on its own labelling rule.

Bars whose forward window does not fit cleanly into any regime bucket
(mixed / noisy) are dropped — same logic as the triple-barrier in the
direction trainer: do not feed ambiguous samples.

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

# Forward window used for labelling. 60 calendar days = a full
# trend/correction cycle in crypto without being so long that the regime
# label tells you nothing about the next move.
FORWARD_HORIZON_DAYS = 60

# Outcome thresholds for forward-window labelling. Tuned to BTC/ETH
# historical magnitude — small enough that labels are populated, large
# enough that "BULL" actually means a real up-move.
_T_BULL_RETURN = 0.10   # forward_return must exceed +10%
_T_BEAR_RETURN = -0.10  # forward_return must drop below -10%
_T_RANGE_RETURN = 0.05  # |forward_return| < 5% to be ACCUMULATION
_T_DRAWDOWN_BULL = -0.05   # bull-trending bars tolerate at most -5% pullback
_T_RUNUP_BEAR = 0.05       # bear-trending bars tolerate at most +5% rally
_T_RUNUP_CORRECTION = 0.05 # correction bars saw a meaningful runup
_T_DRAWDOWN_CORRECTION = -0.10  # ... AND a deeper pullback
_T_RUNUP_RANGE = 0.10
_T_DRAWDOWN_RANGE = -0.10


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


def forward_labels(close: pd.Series, horizon_bars: int) -> np.ndarray:
    """Forward-realized regime labels.

    Walks ``horizon_bars`` ahead of each bar, measures the eventual
    return, the deepest drawdown, and the highest runup, then buckets:

      BULL_TRENDING   — forward return > +10% with at most -5% drawdown
      BEAR_TRENDING   — forward return < -10% with at most +5% rally
      BULL_CORRECTION — forward runup > +5% AND drawdown < -10%
                        (volatile uptrend with a deep dip)
      ACCUMULATION    — forward return small (-5% to +5%) and contained
                        runup/drawdown (within ±10%)
      else            — mixed/noise → label as NaN, dropped from training

    Returns a string array; NaN appears as the literal string "" so we
    can mask it out before fitting.
    """
    n = len(close)
    labels = np.array([""] * n, dtype=object)
    c_arr = close.to_numpy()
    for i in range(n - horizon_bars):
        ref = c_arr[i]
        if not np.isfinite(ref) or ref <= 0:
            continue
        window = c_arr[i + 1 : i + 1 + horizon_bars]
        end = window[-1]
        fwd_return = end / ref - 1.0
        fwd_runup = window.max() / ref - 1.0
        fwd_drawdown = window.min() / ref - 1.0

        if fwd_return > _T_BULL_RETURN and fwd_drawdown > _T_DRAWDOWN_BULL:
            labels[i] = "BULL_TRENDING"
        elif fwd_return < _T_BEAR_RETURN and fwd_runup < _T_RUNUP_BEAR:
            labels[i] = "BEAR_TRENDING"
        elif fwd_runup > _T_RUNUP_CORRECTION and fwd_drawdown < _T_DRAWDOWN_CORRECTION:
            labels[i] = "BULL_CORRECTION"
        elif (
            abs(fwd_return) < _T_RANGE_RETURN
            and fwd_runup < _T_RUNUP_RANGE
            and fwd_drawdown > _T_DRAWDOWN_RANGE
        ):
            labels[i] = "ACCUMULATION"
    return labels


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

    horizon_bars = FORWARD_HORIZON_DAYS * _CPD[timeframe]
    df["label"] = forward_labels(df["close"], horizon_bars)
    df = df[df["label"] != ""].copy()
    df = df.dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].to_numpy()
    y = df["label"].to_numpy()

    print(
        f"  Training set: {len(X):,} samples  "
        f"(forward window={FORWARD_HORIZON_DAYS}d = {horizon_bars} bars)"
    )
    print("  Label distribution:")
    for label, count in zip(*np.unique(y, return_counts=True), strict=False):
        print(f"    {label}: {count} ({count / len(y):.1%})")
    print()

    if len(X) < 200:
        print("  WARNING: fewer than 200 labelled samples — refusing to fit.")
        return

    tscv = TimeSeriesSplit(n_splits=4)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        mdl = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight="balanced", random_state=42,
        )
        mdl.fit(X[tr_idx], y[tr_idx])
        preds = mdl.predict(X[val_idx])
        print(f"Fold {fold + 1}:")
        print(classification_report(y[val_idx], preds, zero_division=0))

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
