#!/usr/bin/env python3
"""F: Fit DBSCAN key S/R levels (A3).

Clusters daily price highs and lows into support/resistance zones.
Not a trained model — outputs a JSON cache. Rerun weekly.

Usage:
    python scripts/fit_key_levels.py
    python scripts/fit_key_levels.py --csv data/btcusdt_1d.csv --lookback 365
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      default="data/btcusdt_1d.csv")
    parser.add_argument("--out",      default="models/key_levels_cache.json")
    parser.add_argument("--lookback", type=int, default=365, help="Days of history to use")
    parser.add_argument("--eps-pct",  type=float, default=0.003, help="Cluster radius as fraction of price")
    args = parser.parse_args()

    print(f"Loading {args.csv} …")
    df = pd.read_csv(args.csv)
    df = df.sort_values("timestamp").tail(args.lookback).reset_index(drop=True)
    print(f"  {len(df)} daily rows (last {args.lookback} days)")

    current_price = float(df["close"].iloc[-1])
    eps = current_price * args.eps_pct

    # Collect all significant price points
    prices = np.concatenate([df["high"].values, df["low"].values])
    # Add OHLC midpoints as additional anchors
    prices = np.concatenate([prices, ((df["high"] + df["low"]) / 2).values])
    prices = prices.reshape(-1, 1)

    db = DBSCAN(eps=eps, min_samples=3).fit(prices)

    clusters: dict[int, list[float]] = {}
    for label, price in zip(db.labels_, prices.flatten()):
        if label == -1:  # noise point
            continue
        clusters.setdefault(label, []).append(price)

    levels = []
    for cluster_prices in clusters.values():
        arr    = np.array(cluster_prices)
        center = float(np.mean(arr))
        levels.append({
            "center":  round(center, 2),
            "low":     round(float(np.min(arr)), 2),
            "high":    round(float(np.max(arr)), 2),
            "touches": len(cluster_prices),
            "type":    "resistance" if center > current_price else "support",
            "dist_pct": round((center - current_price) / current_price * 100, 2),
        })

    # Sort by touch count (strongest levels first)
    levels.sort(key=lambda x: (-x["touches"], abs(x["dist_pct"])))

    out = {"current_price": current_price, "levels": levels[:15]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFound {len(levels)} clusters. Top levels:")
    for lvl in levels[:8]:
        marker = " ◄ current price" if abs(lvl["dist_pct"]) < 1.5 else ""
        print(f"  {lvl['type']:>10}  ${lvl['center']:>10,.0f}  "
              f"({lvl['touches']} touches, {lvl['dist_pct']:+.1f}%){marker}")

    print(f"\nSaved {min(15, len(levels))} levels → {args.out}")


if __name__ == "__main__":
    main()
