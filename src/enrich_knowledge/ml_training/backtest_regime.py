"""Walk-forward backtest for the Random Forest regime classifier.

The RF is trained against rule-based labels (`label_regime` in
`regime.py`), so RF-vs-rule agreement is partly circular. Three
questions remain that the rolling backtest can actually answer:

1. **Transition lag** — when the rule flips from regime A to regime B
   on day X, how many days does the RF need to follow? Answer is
   non-zero because the RF is trained on past rule decisions.

2. **Stability** — RF predictions per month vs rule transitions per
   month. Higher RF flap rate = noisy gating; lower = useful smoothing.

3. **Economic validity** — for each predicted regime, what is the mean
   forward 30-day return? If the four labels produce indistinguishable
   forward-return distributions, the regime feature carries no
   information for the LLM regardless of how well it tracks the rule.

Question 3 is the only non-circular check on whether the labels mean
anything in the market.

Usage:
    python -m src.enrich_knowledge.ml_training.backtest_regime --symbol btcusdt
    python -m src.enrich_knowledge.ml_training.backtest_regime --symbol ethusdt --refit-every-days 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .regime import FEATURE_COLS, compute_features, label_regime

logger = logging.getLogger(__name__)


def _build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def backtest(
    *,
    symbol: str,
    lookback_days: int = 365,
    refit_every_days: int = 30,
    fwd_horizon_days: int = 30,
    csv: str | None = None,
    out: str | None = None,
) -> dict[str, float]:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_1d.csv"
    out_path = out or f"models/1d/backtest_regime_{sym}.csv"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} daily rows loaded")

    df = compute_features(df, timeframe="1d")
    df["rule_label"] = df.apply(label_regime, axis=1)
    df["fwd_ret"] = df["close"].shift(-fwd_horizon_days) / df["close"] - 1
    df = df.dropna(subset=FEATURE_COLS + ["rule_label"]).reset_index(drop=True)

    if len(df) < lookback_days + refit_every_days:
        raise SystemExit(
            f"Not enough data: {len(df):,} rows < lookback {lookback_days} "
            f"+ refit_step {refit_every_days}"
        )

    n_steps = (len(df) - lookback_days) // refit_every_days
    print(
        f"  Walk-forward: {n_steps} folds, refit every {refit_every_days}d "
        f"on rolling {lookback_days}d window  [REGIME]"
    )
    print(f"  Forward-return horizon: {fwd_horizon_days}d\n")

    pred_chunks: list[np.ndarray] = []
    rule_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []

    for i in range(n_steps):
        train_start = i * refit_every_days
        train_end = train_start + lookback_days
        test_end = min(train_end + refit_every_days, len(df))

        train = df.iloc[train_start:train_end]
        test = df.iloc[train_end:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        if len(set(train["rule_label"])) < 2:
            continue

        model = _build_model()
        model.fit(train[FEATURE_COLS].values, train["rule_label"].values)
        pred = model.predict(test[FEATURE_COLS].values)

        pred_chunks.append(pred)
        rule_chunks.append(test["rule_label"].to_numpy())
        fwd_chunks.append(test["fwd_ret"].to_numpy())
        ts_chunks.append(test["timestamp"].values)

        if (i + 1) % 10 == 0 or i == n_steps - 1:
            d0 = pd.Timestamp(test.iloc[0]["timestamp"]).date()
            agree = (pred == test["rule_label"].to_numpy()).mean()
            print(f"  fold {i + 1:>3}/{n_steps}  start {d0}  agreement={agree:.3f}")

    pred_all = np.concatenate(pred_chunks)
    rule_all = np.concatenate(rule_chunks)
    fwd_all = np.concatenate(fwd_chunks)
    ts_all = np.concatenate(ts_chunks)

    agreement = (pred_all == rule_all).mean()

    # 1. Transition lag — when rule changes label, how many days until pred follows?
    lags: list[int] = []
    rule_transitions = np.where(rule_all[1:] != rule_all[:-1])[0] + 1
    for idx in rule_transitions:
        new_label = rule_all[idx]
        following = pred_all[idx:]
        match = np.where(following == new_label)[0]
        if len(match) > 0:
            lags.append(int(match[0]))
    lag_arr = np.array(lags) if lags else np.array([np.nan])

    # 2. Stability — flips per 30 days
    rule_flips = (rule_all[1:] != rule_all[:-1]).sum()
    pred_flips = (pred_all[1:] != pred_all[:-1]).sum()
    days = len(pred_all)
    rule_flip_rate = rule_flips / days * 30
    pred_flip_rate = pred_flips / days * 30

    # 3. Economic validity — forward-return distribution per predicted regime
    print("\n=== Walk-forward summary [REGIME] ===")
    print(f"  Out-of-sample days:        {len(pred_all):,}")
    print(f"  Agreement with rule:       {agreement:.4f}    (circular — RF trained on rule)")
    print()
    print("  Transition lag (days from rule flip to RF following):")
    print(
        f"    n_transitions={len(lags)}  mean={np.nanmean(lag_arr):.2f}  "
        f"median={np.nanmedian(lag_arr):.1f}  p90={np.nanpercentile(lag_arr, 90):.1f}  "
        f"max={np.nanmax(lag_arr):.0f}"
    )
    print()
    print("  Stability (label flips per 30 days):")
    print(f"    rule_flips:  {rule_flip_rate:.2f}")
    print(f"    pred_flips:  {pred_flip_rate:.2f}    "
          f"(RF {'smooths' if pred_flips < rule_flips else 'amplifies'} the rule)")
    print()
    print(f"  Economic validity — forward {fwd_horizon_days}d return per predicted regime:")
    print(f"    {'regime':<20} {'n':>6} {'mean':>8} {'median':>8} {'std':>8}")
    summary: dict[str, dict[str, float]] = {}
    for regime in sorted(set(pred_all)):
        mask = (pred_all == regime) & ~np.isnan(fwd_all)
        if mask.sum() == 0:
            continue
        rets = fwd_all[mask]
        summary[regime] = {
            "n": int(mask.sum()),
            "mean": float(rets.mean()),
            "median": float(np.median(rets)),
            "std": float(rets.std()),
        }
        print(f"    {regime:<20} {summary[regime]['n']:>6} "
              f"{summary[regime]['mean']:>+8.4f} "
              f"{summary[regime]['median']:>+8.4f} "
              f"{summary[regime]['std']:>8.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": ts_all,
        "pred": pred_all,
        "rule": rule_all,
        "fwd_ret": fwd_all,
    }).to_csv(out_path, index=False)
    print(f"\nSaved per-day trace -> {out_path}")

    return {
        "agreement": float(agreement),
        "mean_lag_days": float(np.nanmean(lag_arr)) if lags else float("nan"),
        "rule_flips_per_30d": float(rule_flip_rate),
        "pred_flips_per_30d": float(pred_flip_rate),
        "n_predictions": float(len(pred_all)),
        "n_folds": float(n_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--refit-every-days", type=int, default=30)
    parser.add_argument("--fwd-horizon-days", type=int, default=30)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    backtest(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        refit_every_days=args.refit_every_days,
        fwd_horizon_days=args.fwd_horizon_days,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
