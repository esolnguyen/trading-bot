#!/usr/bin/env python3
"""F: Master offline training script.

Checks each model's retraining trigger and runs the appropriate script.
Run this on a schedule (e.g. weekly cron) or manually.

Usage:
    python scripts/retrain_all.py              # check triggers, retrain if needed
    python scripts/retrain_all.py --force      # retrain everything unconditionally
    python scripts/retrain_all.py --dry-run    # show what would run, don't execute
    python scripts/retrain_all.py --auto       # retrain only models older than their
                                               # configured threshold; exit 0 always
                                               # (safe for cron/scheduled tasks)

Cron example (retrain every Sunday at 02:00 UTC):
    0 2 * * 0  cd /path/to/bot && python scripts/retrain_all.py --auto >> logs/retrain.log 2>&1
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _count_csv_rows(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # subtract header


def _count_closed_trades(data_dir: str = "data") -> int:
    total = 0
    for path in glob.glob(f"{data_dir}/trade_history_*.json"):
        with open(path) as f:
            trades = json.load(f)
        if isinstance(trades, list):
            total += sum(1 for t in trades if str(t.get("action", "")).startswith("CLOSE"))
    return total


def _model_age_days(model_path: str) -> float:
    p = Path(model_path)
    if not p.exists():
        return float("inf")
    mtime = p.stat().st_mtime
    return (time.time() - mtime) / 86400


def _run(script: str, dry_run: bool) -> bool:
    cmd = [sys.executable, script]
    print(f"  → Running: {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit {result.returncode})")
        return False
    print(f"  ✓ Done")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true", help="Retrain unconditionally")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, don't execute")
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Cron-safe mode: retrain only stale models (same as default but "
            "always exits 0 and prints a machine-readable summary line)"
        ),
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== retrain_all.py  {now} ===\n")

    tasks = [
        {
            "name":    "Key Levels (DBSCAN A3)",
            "script":  str(base / "fit_key_levels.py"),
            "trigger": _model_age_days("models/key_levels_cache.json") > 7,
            "reason":  f"cache age {_model_age_days('models/key_levels_cache.json'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_1d.csv") >= 100,
        },
        {
            "name":    "Anomaly Detector (IsoForest B4)",
            "script":  str(base / "train_anomaly.py"),
            "trigger": _model_age_days("models/isolation_forest.joblib") > 7,
            "reason":  f"model age {_model_age_days('models/isolation_forest.joblib'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_15m.csv") >= 5_000,
        },
        {
            "name":    "XGBoost Direction (B2)",
            "script":  str(base / "train_direction.py"),
            "trigger": _model_age_days("models/xgboost_direction.joblib") > 7,
            "reason":  f"model age {_model_age_days('models/xgboost_direction.joblib'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_15m.csv") >= 5_000,
        },
        {
            "name":    "Regime Classifier (RF A4)",
            "script":  str(base / "train_regime.py"),
            "trigger": _model_age_days("models/regime_classifier.joblib") > 30,
            "reason":  f"model age {_model_age_days('models/regime_classifier.joblib'):.1f}d > 30d",
            "needs":   _count_csv_rows("data/btcusdt_1d.csv") >= 200,
        },
        {
            "name":    "Outcome Predictor (LogReg B3)",
            "script":  str(base / "train_outcome.py"),
            "trigger": _count_closed_trades() >= 50
                       and _model_age_days("models/outcome_predictor.joblib") > 3,
            "reason":  f"{_count_closed_trades()} closed trades, model age {_model_age_days('models/outcome_predictor.joblib'):.1f}d",
            "needs":   _count_closed_trades() >= 50,
        },
    ]

    ran = 0
    skipped_data = 0
    for task in tasks:
        name    = task["name"]
        trigger = args.force or task["trigger"]
        needs   = task["needs"]

        if not needs:
            print(f"[SKIP] {name} — insufficient data")
            skipped_data += 1
            continue
        if not trigger:
            print(f"[OK]   {name} — up to date ({task['reason']})")
            continue

        print(f"[RUN]  {name} — {task['reason']}")
        ok = _run(task["script"], args.dry_run)
        if ok:
            ran += 1
        print()

    if ran == 0:
        print("\nAll models are up to date. Nothing retrained.")
    else:
        print(f"\n{'Would retrain' if args.dry_run else 'Retrained'} {ran} model(s).")

    if args.dry_run:
        print("\n(dry-run — no files written)")

    # Machine-readable summary for cron log parsing
    if args.auto:
        status = "ok" if ran >= 0 else "error"
        print(
            f"\nAUTO_SUMMARY retrained={ran} skipped_data={skipped_data} status={status} ts={now}"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _count_csv_rows(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # subtract header


def _count_closed_trades(data_dir: str = "data") -> int:
    total = 0
    for path in glob.glob(f"{data_dir}/trade_history_*.json"):
        with open(path) as f:
            trades = json.load(f)
        if isinstance(trades, list):
            total += sum(1 for t in trades if str(t.get("action", "")).startswith("CLOSE"))
    return total


def _model_age_days(model_path: str) -> float:
    p = Path(model_path)
    if not p.exists():
        return float("inf")
    mtime = p.stat().st_mtime
    return (time.time() - mtime) / 86400


def _run(script: str, dry_run: bool) -> bool:
    cmd = [sys.executable, script]
    print(f"  → Running: {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit {result.returncode})")
        return False
    print(f"  ✓ Done")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== retrain_all.py  {now} ===\n")

    tasks = [
        {
            "name":    "Key Levels (DBSCAN A3)",
            "script":  str(base / "fit_key_levels.py"),
            "trigger": _model_age_days("models/key_levels_cache.json") > 7,
            "reason":  f"cache age {_model_age_days('models/key_levels_cache.json'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_1d.csv") >= 100,
        },
        {
            "name":    "Anomaly Detector (IsoForest B4)",
            "script":  str(base / "train_anomaly.py"),
            "trigger": _model_age_days("models/isolation_forest.joblib") > 7,
            "reason":  f"model age {_model_age_days('models/isolation_forest.joblib'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_15m.csv") >= 5_000,
        },
        {
            "name":    "XGBoost Direction (B2)",
            "script":  str(base / "train_direction.py"),
            "trigger": _model_age_days("models/xgboost_direction.joblib") > 7,
            "reason":  f"model age {_model_age_days('models/xgboost_direction.joblib'):.1f}d > 7d",
            "needs":   _count_csv_rows("data/btcusdt_15m.csv") >= 5_000,
        },
        {
            "name":    "Regime Classifier (RF A4)",
            "script":  str(base / "train_regime.py"),
            "trigger": _model_age_days("models/regime_classifier.joblib") > 30,
            "reason":  f"model age {_model_age_days('models/regime_classifier.joblib'):.1f}d > 30d",
            "needs":   _count_csv_rows("data/btcusdt_1d.csv") >= 200,
        },
        {
            "name":    "Outcome Predictor (LogReg B3)",
            "script":  str(base / "train_outcome.py"),
            "trigger": _count_closed_trades() >= 50
                       and _model_age_days("models/outcome_predictor.joblib") > 3,
            "reason":  f"{_count_closed_trades()} closed trades, model age {_model_age_days('models/outcome_predictor.joblib'):.1f}d",
            "needs":   _count_closed_trades() >= 50,
        },
    ]

    ran = 0
    for task in tasks:
        name    = task["name"]
        trigger = args.force or task["trigger"]
        needs   = task["needs"]

        if not needs:
            print(f"[SKIP] {name} — insufficient data")
            continue
        if not trigger:
            print(f"[OK]   {name} — up to date ({task['reason']})")
            continue

        print(f"[RUN]  {name} — {task['reason']}")
        ok = _run(task["script"], args.dry_run)
        if ok:
            ran += 1
        print()

    if ran == 0:
        print("\nAll models are up to date. Nothing retrained.")
    else:
        print(f"\n{'Would retrain' if args.dry_run else 'Retrained'} {ran} model(s).")

    if args.dry_run:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()
