"""Driver: DBSCAN-based key-level cache (support/resistance).

Clusters daily price highs/lows/midpoints into S/R zones and writes a
JSON cache per symbol. Not a trained model — rerun weekly.

Usage:
    python -m src.enrich_knowledge.runners.run_training --model key_levels
    python -m src.enrich_knowledge.ml_training.key_levels --symbol btcusdt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.enrich_knowledge.config import EnrichKnowledgeSettings

logger = logging.getLogger(__name__)


def fit(
    *,
    symbol: str,
    csv: str | None = None,
    out: str | None = None,
    lookback: int = 365,
    # 0.5% chosen over the original 0.3% after walk-forward backtesting
    # showed eps=0.003 packs ~15 levels so densely that ~half of all
    # daily candles sit within ±2% of one (lift ≈ 1.0 — no edge). At
    # eps=0.005 + 0.5% touch zone, ETH pooled lift jumps to 1.64.
    # See docs/ml-backtest.md for the full run.
    eps_pct: float = 0.005,
) -> None:
    sym = symbol.lower()
    csv_path = csv or f"data/ohlcv/{sym}_1d.csv"
    out_path = out or f"models/1d/key_levels_{sym}_cache.json"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").tail(lookback).reset_index(drop=True)
    print(f"  {len(df)} daily rows (last {lookback} days)")

    current_price = float(df["close"].iloc[-1])
    eps = current_price * eps_pct

    prices = np.concatenate([df["high"].values, df["low"].values])
    prices = np.concatenate([prices, ((df["high"] + df["low"]) / 2).values])
    prices = prices.reshape(-1, 1)

    db = DBSCAN(eps=eps, min_samples=3).fit(prices)

    clusters: dict[int, list[float]] = {}
    for label, price in zip(db.labels_, prices.flatten(), strict=False):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(price)

    levels = []
    for cluster_prices in clusters.values():
        arr = np.array(cluster_prices)
        center = float(np.mean(arr))
        levels.append({
            "center": round(center, 2),
            "low": round(float(np.min(arr)), 2),
            "high": round(float(np.max(arr)), 2),
            "touches": len(cluster_prices),
            "type": "resistance" if center > current_price else "support",
            "dist_pct": round((center - current_price) / current_price * 100, 2),
        })

    levels.sort(key=lambda x: (-x["touches"], abs(x["dist_pct"])))

    payload = {"current_price": current_price, "levels": levels[:15]}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nFound {len(levels)} clusters. Top levels:")
    for lvl in levels[:8]:
        marker = " <- current price" if abs(lvl["dist_pct"]) < 1.5 else ""
        print(
            f"  {lvl['type']:>10}  ${lvl['center']:>10,.0f}  "
            f"({lvl['touches']} touches, {lvl['dist_pct']:+.1f}%){marker}"
        )

    print(f"\nSaved {min(15, len(levels))} levels -> {out_path}")


def train(settings: EnrichKnowledgeSettings, dry_run: bool = False) -> None:
    for symbol in settings.ml_training.training_symbols:
        if dry_run:
            logger.info("[dry-run] would fit key_levels %s", symbol)
            continue
        try:
            fit(symbol=symbol)
        except Exception:
            logger.exception("key_levels fit failed for %s", symbol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--eps-pct", type=float, default=0.005)
    args = parser.parse_args()

    fit(
        symbol=args.symbol,
        csv=args.csv,
        out=args.out,
        lookback=args.lookback,
        eps_pct=args.eps_pct,
    )


if __name__ == "__main__":
    main()
