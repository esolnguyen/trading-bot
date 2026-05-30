"""Driver: Logistic Regression trade-outcome predictor.

Used as a hard hit-rate prior on a proposed trade entry. Inference lives
in ``OutcomePredictor`` (mcp_servers/ml_mcp/services/outcome_predictor.py)
and expects exactly six categorical/continuous features:

  rsi, adx, atr_pct, trend, vol_state, bb_pos

``trend``/``vol_state``/``bb_pos`` are bucketed at inference time
(BULLISH→1, HIGH→1, UPPER→1, etc). Training MUST mirror that bucketing
or the model learns one feature space and predicts in another.

Labels: triple-barrier on a long entry — TP hit before SL within
MAX_HORIZON bars at ±BARRIER_K * ATR. Bars where neither barrier is
reached (timeout) are dropped. Pooled across configured training symbols
and timeframes — the inference service loads a single
``models/outcome_predictor.joblib`` regardless of symbol.

Best-practice stack inherited from ``direction.py``:
  * López de Prado sample uniqueness weighting (AFML ch. 4)
  * Purged k-fold + embargo (AFML ch. 7)

Usage:
    python -m src.enrich_knowledge.runners.run_training --model outcome
    python -m src.enrich_knowledge.ml_training.outcome --timeframe 4h
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.enrich_knowledge.ml_training.direction import (
    BARRIER_K,
    MAX_HORIZON,
    RECENCY_LAMBDA,
    _purged_splits,
    compute_features,
    recency_weights,
    sample_uniqueness,
    triple_barrier_labels,
)
from src.features.outcome import (
    OUTCOME_FEATURE_COLS,
    bucket_bb_pos,
    bucket_trend,
    bucket_vol_state,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 720

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}

# Canonical feature list + bucketing thresholds are shared with the live
# gate (src.features.outcome) so train/serve cannot drift.
FEATURE_COLS = OUTCOME_FEATURE_COLS


def _build_training_frame(
    symbol: str, timeframe: str, lookback_days: int
) -> pd.DataFrame | None:
    sym = symbol.lower()
    csv_path = f"data/ohlcv/{sym}_{timeframe}.csv"
    if not Path(csv_path).exists():
        logger.info("outcome: %s missing — skipping", csv_path)
        return None

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    cap = lookback_days * _CANDLES_PER_DAY.get(timeframe, 96)
    if len(df) > cap:
        df = df.tail(cap).reset_index(drop=True)

    df = compute_features(df)
    labels, t1 = triple_barrier_labels(
        df["close"], df["atr"], k=BARRIER_K, max_horizon=MAX_HORIZON
    )
    df["target"] = labels
    df["t1_local"] = t1

    df["rsi"] = df["rsi_14"]
    df["atr_pct"] = df["atr"] / df["close"] * 100
    df["trend"] = bucket_trend(df["ema_spread"], df["adx"])
    df["vol_state"] = bucket_vol_state(df["vol_ratio"])
    df["bb_pos"] = bucket_bb_pos(df["bb_pos"])
    df["symbol"] = symbol.upper()
    return df


def fit(
    *,
    symbols: list[str],
    timeframes: list[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    out: str | None = None,
) -> None:
    out_path = out or "models/outcome_predictor.joblib"

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for timeframe in timeframes:
            frame = _build_training_frame(symbol, timeframe, lookback_days)
            if frame is None or frame.empty:
                continue
            frame = frame.dropna(subset=FEATURE_COLS + ["target"])
            print(f"  {symbol.upper()} {timeframe}: {len(frame):,} labelled bars")
            frames.append(
                frame[FEATURE_COLS + ["target", "t1_local", "timestamp", "symbol"]]
            )

    if not frames:
        print("outcome: no labelled samples available — skipping fit.")
        return

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("timestamp").reset_index(drop=True)

    # Re-project per-symbol t1_local → pooled-row positions, same
    # technique as direction.py's pooler.
    pooled_t1 = np.full(len(pooled), np.nan, dtype=float)
    for sym, group in pooled.groupby("symbol"):
        positions = group.index.to_numpy()
        local_t1 = group["t1_local"].to_numpy()
        for k_, pos in enumerate(positions):
            lt = local_t1[k_]
            if np.isnan(lt):
                continue
            offset = int(lt) - k_
            target_local_idx = k_ + offset
            if target_local_idx < len(positions):
                pooled_t1[pos] = positions[target_local_idx]
    pooled["t1"] = pooled_t1

    # Compute uniqueness BEFORE dropna so t1 (pre-dropna positions) is
    # consistent with index_pos. The weight column rides through dropna
    # with its sample.
    pooled["original_pos"] = np.arange(len(pooled), dtype=float)
    pooled["weight"] = sample_uniqueness(
        pooled["t1"].to_numpy(),
        index_pos=pooled["original_pos"].to_numpy().astype(int),
    )

    pre_drop = len(pooled)
    pooled = pooled.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    print(
        f"\nPooled training set: {len(pooled):,}/{pre_drop:,} labelled samples "
        f"across {len(symbols)} symbol(s) x {len(timeframes)} timeframe(s)"
    )

    if len(pooled) < 500:
        print("outcome: fewer than 500 pooled samples — refusing to fit.")
        return

    X = pooled[FEATURE_COLS].to_numpy()
    y = pooled["target"].astype(int).to_numpy()
    t1 = pooled["t1"].to_numpy()
    original_pos = pooled["original_pos"].to_numpy()

    raw_w = pooled["weight"].to_numpy()
    pos_min = raw_w[raw_w > 0].min() if (raw_w > 0).any() else 1.0
    uniqueness = np.where(raw_w > 0, raw_w, pos_min)
    # CV folds: uniqueness only (recency on per-fold training starves
    # early folds whose train is the old data). Final deployment fit:
    # uniqueness × recency, so the live model leans on recent regime.
    cv_weights = uniqueness
    recency = recency_weights(original_pos)
    final_weights = uniqueness * recency

    win_rate = y.mean()
    print(
        f"  Win rate (TP-before-SL): {win_rate:.1%}\n"
        f"  Mean uniqueness={uniqueness.mean():.3f}  "
        f"mean recency={recency.mean():.3f}  "
        f"final-fit ESS≈{(final_weights.sum() ** 2 / (final_weights ** 2).sum()):.0f}\n"
    )

    aucs: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(
        _purged_splits(
            len(X),
            n_splits=5,
            t1=t1,
            original_pos=original_pos,
            embargo=MAX_HORIZON,
        )
    ):
        if len(tr_idx) < 200 or len(val_idx) < 50:
            print(f"Fold {fold + 1}: too small after purging — skip")
            continue
        mdl = LogisticRegression(
            class_weight="balanced", max_iter=300, random_state=42
        )
        mdl.fit(X[tr_idx], y[tr_idx], sample_weight=cv_weights[tr_idx])
        proba = mdl.predict_proba(X[val_idx])[:, 1]
        preds = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y[val_idx], proba)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"Fold {fold + 1}  AUC={auc:.3f}  (train n={len(tr_idx):,}, val n={len(val_idx):,})")
        print(classification_report(y[val_idx], preds, target_names=["loss", "win"], zero_division=0))

    finite = [a for a in aucs if np.isfinite(a)]
    if finite:
        print(f"Mean walk-forward AUC: {np.mean(finite):.3f} ± {np.std(finite):.3f}\n")

    final = LogisticRegression(
        class_weight="balanced", max_iter=300, random_state=42
    )
    final.fit(X, y, sample_weight=final_weights)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "feature_cols": FEATURE_COLS,
            "max_horizon": MAX_HORIZON,
            "barrier_k": BARRIER_K,
            "recency_lambda": RECENCY_LAMBDA,
        },
        out_path,
    )
    print(f"Saved → {out_path}  ({len(X):,} pooled samples)")


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    symbols = list(settings.ml_training.training_symbols)
    timeframes = list(settings.ml_training.ml_timeframes)
    if dry_run:
        logger.info(
            "[dry-run] would fit outcome on %d symbol(s) x %d timeframe(s)",
            len(symbols), len(timeframes),
        )
        return
    try:
        fit(symbols=symbols, timeframes=timeframes)
    except Exception:
        logger.exception("outcome fit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe",
        default="4h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Single timeframe to fit on (ad-hoc). For full training use the runner.",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT",
        help="Comma-separated symbol list (default: pool all 4 majors).",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fit(
        symbols=symbols,
        timeframes=[args.timeframe],
        lookback_days=args.lookback_days,
        out=args.out,
    )


if __name__ == "__main__":
    main()
