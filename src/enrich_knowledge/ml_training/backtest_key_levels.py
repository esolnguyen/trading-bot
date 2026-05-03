"""Walk-forward backtest for the DBSCAN key-level detector.

Refits the level cache every ``refit_every_days`` on a rolling
``lookback_days`` window of daily candles, then asks: do **forward
reversal points** (local highs/lows over the next ``forward_days``)
cluster near the identified levels more than random candles do?

The test is fully out-of-sample — levels are computed from history
the reversal points are not part of.

Metrics:
  * P(reversal near level)  — fraction of forward reversals whose
    nearest level is within ``near_pct`` of price.
  * P(any candle near level) — same fraction over **every** close
    in the same window. Acts as the baseline (price naturally spends
    time near levels regardless of reversals).
  * Lift = reversal-rate / baseline-rate. > 1 means reversals cluster
    near levels beyond what random price walks would produce.

Lift is the answer to "are these S/R levels real or coincidence?"

Usage:
    python -m src.enrich_knowledge.ml_training.backtest_key_levels --symbol btcusdt
    python -m src.enrich_knowledge.ml_training.backtest_key_levels --symbol ethusdt --near-pct 0.015 --forward-days 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def _detect_levels(df: pd.DataFrame, eps_pct: float = 0.003, top_n: int = 15) -> np.ndarray:
    """Match key_levels.fit clustering, return centers as ndarray."""
    current_price = float(df["close"].iloc[-1])
    eps = current_price * eps_pct
    prices = np.concatenate([
        df["high"].values,
        df["low"].values,
        ((df["high"] + df["low"]) / 2).values,
    ]).reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=3).fit(prices)
    clusters: dict[int, list[float]] = {}
    for label, p in zip(db.labels_, prices.flatten(), strict=False):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(p)
    rows = sorted(
        ({"center": float(np.mean(v)), "touches": len(v)} for v in clusters.values()),
        key=lambda r: -r["touches"],
    )
    return np.array([r["center"] for r in rows[:top_n]])


def _local_extrema(window: pd.DataFrame, span: int = 5) -> np.ndarray:
    """Local highs and lows over a centered ``span`` window. Returns the
    raw extreme prices (highs from h column, lows from l column)."""
    h = window["high"].to_numpy()
    low = window["low"].to_numpy()
    n = len(h)
    half = span // 2
    extrema: list[float] = []
    for i in range(half, n - half):
        sl = slice(i - half, i + half + 1)
        if h[i] == h[sl].max() and h[i] != h[sl].min():
            extrema.append(h[i])
        if low[i] == low[sl].min() and low[i] != low[sl].max():
            extrema.append(low[i])
    return np.array(extrema)


def _frac_near_level(prices: np.ndarray, levels: np.ndarray, near_pct: float) -> float:
    if len(prices) == 0 or len(levels) == 0:
        return float("nan")
    # |price - nearest_level| / price for every price
    diffs = np.abs(prices[:, None] - levels[None, :])
    nearest = diffs.min(axis=1)
    return float((nearest / prices <= near_pct).mean())


def backtest(
    *,
    symbol: str,
    lookback_days: int = 365,
    forward_days: int = 30,
    refit_every_days: int = 30,
    near_pct: float = 0.02,
    eps_pct: float = 0.003,
    extrema_span: int = 5,
    csv: str | None = None,
    out: str | None = None,
) -> dict[str, float]:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_1d.csv"
    out_path = out or f"models/1d/backtest_key_levels_{sym}.csv"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} daily rows loaded")

    if len(df) < lookback_days + forward_days:
        raise SystemExit(
            f"Not enough data: {len(df):,} rows < lookback {lookback_days} "
            f"+ forward {forward_days}"
        )

    n_steps = (len(df) - lookback_days - forward_days) // refit_every_days + 1
    print(
        f"  Walk-forward: {n_steps} folds, refit every {refit_every_days}d  [KEY-LEVELS]"
    )
    print(
        f"  near_pct={near_pct:.1%}  forward window={forward_days}d  "
        f"local-extrema span={extrema_span}\n"
    )

    rev_hits: list[int] = []
    rev_total = 0
    base_hits: list[int] = []
    base_total = 0
    fold_lifts: list[float] = []
    rows: list[dict[str, float | str]] = []

    for i in range(n_steps):
        end = lookback_days + i * refit_every_days
        if end + forward_days > len(df):
            break
        history = df.iloc[end - lookback_days : end]
        future = df.iloc[end : end + forward_days]

        levels = _detect_levels(history, eps_pct=eps_pct)
        if len(levels) == 0:
            continue

        extrema = _local_extrema(future, span=extrema_span)
        all_close = future["close"].to_numpy()

        rev_rate = _frac_near_level(extrema, levels, near_pct)
        base_rate = _frac_near_level(all_close, levels, near_pct)

        n_rev = len(extrema)
        n_base = len(all_close)
        if not np.isnan(rev_rate):
            rev_hits.append(int(round(rev_rate * n_rev)))
            rev_total += n_rev
        base_hits.append(int(round(base_rate * n_base)))
        base_total += n_base

        fold_lift = rev_rate / base_rate if base_rate > 0 else float("nan")
        fold_lifts.append(fold_lift)

        rows.append({
            "fold_end": str(future.iloc[0]["timestamp"].date()),
            "n_levels": len(levels),
            "n_extrema": n_rev,
            "rev_rate": rev_rate,
            "base_rate": base_rate,
            "fold_lift": fold_lift,
        })

        if (i + 1) % 10 == 0 or i == n_steps - 1:
            d0 = pd.Timestamp(future.iloc[0]["timestamp"]).date()
            print(
                f"  fold {i + 1:>3}/{n_steps}  start {d0}  "
                f"levels={len(levels)}  extrema={n_rev}  "
                f"rev_rate={rev_rate:.3f}  base={base_rate:.3f}  lift={fold_lift:.2f}"
            )

    pooled_rev = sum(rev_hits) / rev_total if rev_total else float("nan")
    pooled_base = sum(base_hits) / base_total if base_total else float("nan")
    pooled_lift = pooled_rev / pooled_base if pooled_base else float("nan")
    fold_arr = np.array([x for x in fold_lifts if not np.isnan(x)])

    verdict = (
        "predictive" if pooled_lift > 1.15
        else "no edge" if pooled_lift < 1.05
        else "marginal"
    )

    print("\n=== Walk-forward summary [KEY-LEVELS] ===")
    print(f"  Folds:                       {len(rows)}")
    print(f"  Total reversals scored:      {rev_total:,}")
    print(f"  Total candles in baseline:   {base_total:,}")
    print(f"  P(reversal near level):      {pooled_rev:.4f}")
    print(f"  P(any candle near level):    {pooled_base:.4f}  (baseline)")
    print(f"  Pooled lift:                 {pooled_lift:.3f}    [{verdict}]")
    print(
        f"  Per-fold lift:               mean={fold_arr.mean():.3f}  "
        f"std={fold_arr.std():.3f}  "
        f"min={fold_arr.min():.3f}  max={fold_arr.max():.3f}"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved per-fold trace -> {out_path}")

    return {
        "pooled_lift": float(pooled_lift),
        "pooled_rev_rate": float(pooled_rev),
        "pooled_base_rate": float(pooled_base),
        "n_folds": float(len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--forward-days", type=int, default=30)
    parser.add_argument("--refit-every-days", type=int, default=30)
    parser.add_argument("--near-pct", type=float, default=0.02)
    parser.add_argument("--eps-pct", type=float, default=0.003)
    parser.add_argument("--extrema-span", type=int, default=5)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    backtest(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        forward_days=args.forward_days,
        refit_every_days=args.refit_every_days,
        near_pct=args.near_pct,
        eps_pct=args.eps_pct,
        extrema_span=args.extrema_span,
        csv=args.csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
