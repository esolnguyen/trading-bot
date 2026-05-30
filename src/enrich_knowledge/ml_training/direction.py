"""Driver: XGBoost direction classifier.

Implements the López de Prado (Advances in Financial Machine Learning,
2018) best-practice stack for short-horizon financial direction:

  * Triple-barrier labelling on ±k·ATR with explicit barrier-touch times
  * **Cross-symbol pooling** so every fit sees BTC+ETH+SOL+BNB samples,
    with a categorical `symbol_id` feature so the model can specialise
  * **Sample uniqueness weighting** (AFML ch. 4): bars whose forward
    barrier window overlaps few neighbours get more weight, bars in
    crowded windows get less. This corrects the IID assumption that
    classifiers make on serially overlapping labels.
  * **Purged k-fold + embargo** (AFML ch. 7): drop train rows whose
    barrier-touch time t1 falls inside the validation fold AND drop the
    last MAX_HORIZON rows of train as a hard embargo. Prevents the
    forward-window leak that inflates naive TimeSeriesSplit AUCs.
  * **Isotonic calibration** so `predict_proba` returns numbers that
    actually mean what the trade-skill prompt claims they mean.

Output: one pooled model per timeframe at
``models/<tf>/xgboost_direction.joblib``. Per-symbol models are no
longer produced — the inference service falls back to the global model
and keys on the new `symbol_id` feature.

Usage:
    python -m src.enrich_knowledge.runners.run_training --model direction
    python -m src.enrich_knowledge.ml_training.direction --timeframe 4h
    python -m src.enrich_knowledge.ml_training.direction --timeframe 4h --symbol btcusdt   # ad-hoc single fit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.enrich_knowledge.config import EnrichKnowledgeSettings
from src.features.obv import (
    OBV_SLOPE_PERIOD,
    rolling_normalized_obv_slope,
)

logger = logging.getLogger(__name__)

# Triple-barrier window. 24 bars on 4h ≈ 4 calendar days; on 1h ≈ 24h.
# The previous 6-bar window timed out 57% of bars on ETHUSDT 4h —
# stretching the window keeps far more samples labelled. Barrier at
# 1.5×ATR (was 1.0) demands a slightly stronger move, which the larger
# window can produce.
MAX_HORIZON = 24
BARRIER_K = 1.5

# Drop the prior 180d cap. With pooled training (4 symbols × full
# history) and `realized_vol`-style features we want the model to learn
# regime-conditional patterns rather than restrict to one regime.
DEFAULT_LOOKBACK_DAYS = 720

# Exponential time-decay applied per-sample on top of López de Prado
# uniqueness weights. lambda=1.0 means the oldest bar in the training
# set carries ~exp(-1) ≈ 0.37× the weight of the newest. Crypto regimes
# turn over on a months timescale, so recent samples are more
# representative of the live distribution. Set to 0.0 to disable.
RECENCY_LAMBDA = 1.0

_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
}

# `symbol_id` is the last entry — added at training so the inference
# service can append it from the per-bundle symbol_map without breaking
# feature-order assumptions.
FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "ema_spread", "ema_20_dist", "ema_50_dist",
    "atr_pct", "adx", "bb_pos", "obv_slope",
    "vol_ratio", "choppiness", "cci_14",
    "symbol_id",
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

    df["obv_slope"] = rolling_normalized_obv_slope(
        c.to_numpy(), v.to_numpy(), OBV_SLOPE_PERIOD
    )

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

    return df


def triple_barrier_labels(
    close: pd.Series,
    atr: pd.Series,
    *,
    k: float = BARRIER_K,
    max_horizon: int = MAX_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Triple-barrier labels + barrier-touch indices.

    For each bar i, scan the next ``max_horizon`` bars and record:
      label=1.0, t1=j  if close[j] >= ref + k*atr  (upper barrier hit at j)
      label=0.0, t1=j  if close[j] <= ref - k*atr  (lower barrier hit at j)
      label=NaN, t1=NaN if neither hits within the horizon (timeout)

    Returning ``t1`` lets us:
      - compute López de Prado sample-uniqueness weights, which correct
        the IID assumption sklearn / XGBoost make on serially overlapping
        labels;
      - **purge** training rows whose barrier window leaks into a
        validation fold, instead of just embargoing the last MAX_HORIZON
        rows blindly.
    """
    n = len(close)
    labels = np.full(n, np.nan, dtype=float)
    t1 = np.full(n, np.nan, dtype=float)
    c_arr = close.to_numpy()
    a_arr = atr.to_numpy()
    for i in range(n - max_horizon):
        a_i = a_arr[i]
        if not np.isfinite(a_i) or a_i <= 0:
            continue
        ref = c_arr[i]
        upper = ref + k * a_i
        lower = ref - k * a_i
        for j in range(i + 1, i + 1 + max_horizon):
            cj = c_arr[j]
            if cj >= upper:
                labels[i] = 1.0
                t1[i] = j
                break
            if cj <= lower:
                labels[i] = 0.0
                t1[i] = j
                break
    return labels, t1


def sample_uniqueness(
    t1: np.ndarray, index_pos: np.ndarray | None = None
) -> np.ndarray:
    """López de Prado AFML ch. 4 sample uniqueness.

    ``t1[i]`` is the absolute row position at which sample i's barrier
    closed. Concurrency at time t = number of samples whose barrier
    window covers t. Each sample i contributes uniqueness equal to the
    average ``1 / concurrency(t)`` across t ∈ (i, t1[i]].

    Returns one weight per sample. NaN-target rows get weight 0 — the
    caller must filter them out before fitting.
    """
    n = len(t1)
    if index_pos is None:
        index_pos = np.arange(n, dtype=int)
    valid = ~np.isnan(t1)
    if not np.any(valid):
        return np.zeros(n)
    max_t = int(np.max(t1[valid]))

    # concurrency[t] counts how many samples' window (i_pos, t1] covers t.
    concurrency = np.zeros(max_t + 2, dtype=float)
    for i in range(n):
        if not valid[i]:
            continue
        start = int(index_pos[i]) + 1
        end = int(t1[i])
        if start > end:
            continue
        concurrency[start] += 1
        concurrency[end + 1] -= 1
    concurrency = np.cumsum(concurrency)

    weights = np.zeros(n, dtype=float)
    for i in range(n):
        if not valid[i]:
            continue
        start = int(index_pos[i]) + 1
        end = int(t1[i])
        if start > end:
            continue
        window = concurrency[start : end + 1]
        nonzero = np.where(window > 0, window, 1.0)
        weights[i] = float(np.mean(1.0 / nonzero))
    return weights


def recency_weights(
    original_pos: np.ndarray, lambda_: float = RECENCY_LAMBDA
) -> np.ndarray:
    """Exponential time-decay weights.

    Newest bar gets weight 1.0; oldest gets ``exp(-lambda_)``. Combined
    multiplicatively with sample-uniqueness weights, this addresses the
    crypto-specific issue that the per-fold AUCs show: a model fit on
    the full 720-day window forgets the recent regime by the time the
    last validation fold arrives. Down-weighting old samples lets the
    final fit lean on the most recent regime without throwing the
    history away.
    """
    n = len(original_pos)
    if n == 0:
        return np.zeros(0)
    max_pos = float(np.max(original_pos))
    if max_pos <= 0:
        return np.ones(n)
    age = (max_pos - original_pos.astype(float)) / max_pos
    return np.exp(-lambda_ * age)


def _purged_splits(
    n: int,
    n_splits: int,
    t1: np.ndarray,
    original_pos: np.ndarray,
    embargo: int,
):
    """Walk-forward splits with purging and embargo.

    Both ``t1`` and ``original_pos`` are arrays of length ``n`` (same as
    the post-dropna training matrix). ``original_pos[i]`` is the row
    position of sample i in the *pre-dropna* pooled frame; ``t1[i]`` is
    the barrier-touch position in that same pre-dropna space. The CV
    indices, by contrast, run over post-dropna positions.

    Yields ``(tr_idx, val_idx)`` pairs where tr_idx has been purged of
    any sample whose barrier window ends inside the validation fold's
    pre-dropna span (proper purging), then embargoed by dropping the
    last ``embargo`` rows of train.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, val_idx in tscv.split(np.arange(n)):
        if len(val_idx) == 0 or len(tr_idx) == 0:
            yield tr_idx, val_idx
            continue
        val_start_orig = int(original_pos[val_idx[0]])
        val_end_orig = int(original_pos[val_idx[-1]])
        purge_until = val_end_orig + embargo

        keep = np.ones(len(tr_idx), dtype=bool)
        for k_, idx in enumerate(tr_idx):
            t1_orig = t1[idx]
            if not np.isnan(t1_orig) and val_start_orig <= t1_orig <= purge_until:
                keep[k_] = False
        tr_idx = tr_idx[keep]

        if embargo > 0 and len(tr_idx) > embargo:
            tr_idx = tr_idx[:-embargo]
        yield tr_idx, val_idx


def _build_symbol_frame(
    symbol: str, timeframe: str, lookback_days: int, symbol_id: int
) -> pd.DataFrame | None:
    sym = symbol.lower()
    csv_path = f"data/ohlcv/{sym}_{timeframe}.csv"
    if not Path(csv_path).exists():
        logger.info("direction: %s missing — skipping", csv_path)
        return None

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    cap = lookback_days * _CANDLES_PER_DAY.get(timeframe, 96)
    if len(df) > cap:
        df = df.tail(cap).reset_index(drop=True)

    df = compute_features(df)

    labels, t1 = triple_barrier_labels(df["close"], df["atr"])
    df["target"] = labels
    df["t1_local"] = t1
    df["symbol_id"] = float(symbol_id)
    df["symbol"] = symbol.upper()
    return df


def _pool_training_data(
    symbols: list[str], timeframe: str, lookback_days: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load + label every available symbol for ``timeframe``, concatenate
    by timestamp, and assign each symbol a stable integer id."""
    frames: list[pd.DataFrame] = []
    symbol_map: dict[str, int] = {}
    for idx, symbol in enumerate(sorted({s.upper() for s in symbols})):
        symbol_map[symbol] = idx
        frame = _build_symbol_frame(symbol, timeframe, lookback_days, idx)
        if frame is None:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), symbol_map

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("timestamp").reset_index(drop=True)

    # After concat the per-symbol t1_local is in *that symbol's* index
    # space. Re-project to pooled-row positions: a sample and its
    # barrier window must be in the same symbol's slice. We compute the
    # pooled offset by grouping on `symbol`.
    pooled_t1 = np.full(len(pooled), np.nan, dtype=float)
    # pre-compute, per symbol, the row-positions in pooled order
    for sym, group in pooled.groupby("symbol"):
        positions = group.index.to_numpy()  # pooled-row positions, sorted by timestamp
        local_t1 = group["t1_local"].to_numpy()
        for k_, pos in enumerate(positions):
            lt = local_t1[k_]
            if np.isnan(lt):
                continue
            offset = int(lt) - k_  # how many in-symbol bars forward the touch was
            target_local_idx = k_ + offset
            if target_local_idx < len(positions):
                pooled_t1[pos] = positions[target_local_idx]
    pooled["t1"] = pooled_t1
    pooled = pooled.drop(columns=["t1_local"])
    return pooled, symbol_map


def fit_pooled(
    *,
    symbols: list[str],
    timeframe: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    out: str | None = None,
) -> None:
    out_path = out or f"models/{timeframe}/xgboost_direction.joblib"

    pooled, symbol_map = _pool_training_data(symbols, timeframe, lookback_days)
    if pooled.empty:
        print(f"direction: no symbols available for {timeframe} — skipping")
        return

    # Compute uniqueness BEFORE dropna so t1 (in original-pos space) is
    # consistent with index_pos. The weight column survives the drop
    # alongside the rows it belongs to.
    pooled = pooled.reset_index(drop=True)
    pooled["original_pos"] = np.arange(len(pooled), dtype=float)
    pooled["weight"] = sample_uniqueness(
        pooled["t1"].to_numpy(),
        index_pos=pooled["original_pos"].to_numpy().astype(int),
    )

    n_total = len(pooled)
    pooled = pooled.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    n_used = len(pooled)
    print(
        f"  Pooled across {len(symbol_map)} symbol(s) on {timeframe}: "
        f"{n_used:,}/{n_total:,} bars labelled "
        f"({100 * n_used / max(n_total, 1):.1f}% — timeouts dropped)\n"
    )
    if n_used < 500:
        print("  WARNING: fewer than 500 pooled samples — refusing to fit.")
        return

    X = pooled[FEATURE_COLS].to_numpy()
    y = pooled["target"].astype(int).to_numpy()
    t1 = pooled["t1"].to_numpy()
    original_pos = pooled["original_pos"].to_numpy()

    raw_w = pooled["weight"].to_numpy()
    pos_min = raw_w[raw_w > 0].min() if (raw_w > 0).any() else 1.0
    uniqueness = np.where(raw_w > 0, raw_w, pos_min)
    # CV folds use uniqueness only — recency on per-fold training would
    # starve early folds (their train *is* the old data) and produce
    # degenerate AUCs that misrepresent honest model performance.
    cv_weights = uniqueness
    # Final deployment fit uses uniqueness × recency: we want the live
    # model to lean on the most recent regime, since that's the
    # distribution the next prediction will sample from.
    recency = recency_weights(original_pos)
    final_weights = uniqueness * recency

    bullish_ratio = y.mean()
    scale_pos_weight = (1 - bullish_ratio) / bullish_ratio if bullish_ratio > 0 else 1.0
    print(f"  Symbol map: {symbol_map}")
    print(
        f"  Class balance: {bullish_ratio:.1%} bullish  "
        f"scale_pos_weight={scale_pos_weight:.2f}\n"
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
            print(f"Fold {fold + 1}: too small after purging (tr={len(tr_idx)}, val={len(val_idx)}) — skip")
            continue
        mdl = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", verbosity=0,
        )
        mdl.fit(
            X[tr_idx], y[tr_idx],
            sample_weight=cv_weights[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        proba = mdl.predict_proba(X[val_idx])[:, 1]
        preds = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y[val_idx], proba)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"Fold {fold + 1}  AUC={auc:.3f}  (train n={len(tr_idx):,}, val n={len(val_idx):,})")
        print(classification_report(y[val_idx], preds, target_names=["bearish", "bullish"], zero_division=0))

    finite = [a for a in aucs if np.isfinite(a)]
    if finite:
        print(f"Mean walk-forward AUC: {np.mean(finite):.3f} ± {np.std(finite):.3f}\n")

    base = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", verbosity=0,
    )
    final = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=5),
    )
    final.fit(X, y, sample_weight=final_weights)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "feature_cols": FEATURE_COLS,
            "symbol_map": symbol_map,
            "max_horizon": MAX_HORIZON,
            "barrier_k": BARRIER_K,
            "recency_lambda": RECENCY_LAMBDA,
        },
        out_path,
    )
    print(f"Saved → {out_path}  ({len(X):,} pooled samples)")


def fit(
    *,
    symbol: str,
    timeframe: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    csv: str | None = None,
    out: str | None = None,
) -> None:
    """Single-symbol ad-hoc fit. Kept for fast inspection of one
    (symbol, timeframe) pair — production training goes through
    ``fit_pooled``."""
    out_path = out or f"models/{timeframe}/xgboost_direction_{symbol.lower()}.joblib"
    fit_pooled(
        symbols=[symbol.upper()],
        timeframe=timeframe,
        lookback_days=lookback_days,
        out=out_path,
    )


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    """Production training: one pooled model per timeframe."""
    symbols = list(settings.ml_training.training_symbols)
    for timeframe in settings.ml_training.ml_timeframes:
        if dry_run:
            logger.info(
                "[dry-run] would fit pooled direction on %s across %s",
                timeframe, symbols,
            )
            continue
        try:
            fit_pooled(symbols=symbols, timeframe=timeframe)
        except Exception:
            logger.exception("direction pooled fit failed for %s", timeframe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe",
        default="4h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Single symbol for ad-hoc inspection (default: pool BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT).",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list for pooled fit (overrides --symbol).",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        fit_pooled(
            symbols=symbols,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            out=args.out,
        )
    elif args.symbol:
        fit(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            out=args.out,
        )
    else:
        fit_pooled(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            out=args.out,
        )


if __name__ == "__main__":
    main()
