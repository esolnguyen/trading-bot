#!/usr/bin/env python3
"""E: One-time OHLCV backfill.

Downloads historical candles from Binance and saves them to
data/ohlcv/<symbol>_<interval>.csv. Run this once before training any ML model.

Usage:
    # Fetch the ML timeframe set in .env (ML_TIMEFRAME) + daily for regime:
    python scripts/backfill_ohlcv.py

    # Fetch a specific timeframe:
    python scripts/backfill_ohlcv.py --interval 4h
    python scripts/backfill_ohlcv.py --interval 1h
    python scripts/backfill_ohlcv.py --interval 15m

    # Fetch all three ML timeframes + 1d at once:
    python scripts/backfill_ohlcv.py --all

    # Override symbol or row count:
    python scripts/backfill_ohlcv.py --symbol ETHUSDT --interval 4h --rows 16500
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=False)

from src.core.config import Settings
from src.infrastructure.binance.rest_client import BinanceRestClient

# Default row counts per interval — enough for full Binance history from 2017.
# Training scripts will take a recent slice; having more data never hurts.
_DEFAULT_ROWS: dict[str, int] = {
    "1m":  43_200,   # ~30 days (Binance REST keeps ~30 days of 1m data)
    "5m":  17_280,   # ~60 days
    "15m": 70_000,   # ~2 years
    "1h":  17_520,   # ~2 years
    "4h":  16_500,   # ~full history since 2017
    "1d":  2_800,    # ~full history since 2017
}

_ML_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h")


def _read_last_timestamp(path: Path) -> int | None:
    """Return the last timestamp (ms) already in the CSV, or None if empty/missing."""
    if not path.exists():
        return None
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        # rows[0] is header; find last non-empty data row
        for row in reversed(rows[1:]):
            if row:
                return int(row[0])
    except Exception:
        pass
    return None


def backfill(
    client: BinanceRestClient,
    symbol: str,
    interval: str,
    target_rows: int,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_ts = _read_last_timestamp(path)

    if last_ts is not None:
        # --- APPEND MODE: fetch only candles newer than what we already have ---
        print(f"Updating {symbol} {interval} → {output_path} (last candle: {last_ts})")
        new_rows: list[list] = []
        start_ms = last_ts + 1

        while True:
            kwargs: dict = dict(symbol=symbol, interval=interval, limit=1000, startTime=start_ms)
            try:
                batch = client.get_klines(**kwargs)
            except Exception as exc:
                print(f"  ERROR: {exc} — retrying in 10s")
                time.sleep(10)
                continue

            if not batch:
                break

            # Drop the last candle — it may be the still-open current candle
            batch = batch[:-1]
            if not batch:
                break

            new_rows.extend(batch)
            start_ms = int(batch[-1][0]) + 1
            print(f"  +{len(batch)} rows  newest={batch[-1][0]}")
            time.sleep(0.12)

        if not new_rows:
            print("  Already up to date.\n")
            return

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            for row in new_rows:
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])

        print(f"Done. Appended {len(new_rows):,} new rows → {path}\n")

    else:
        # --- FULL BACKFILL: walk backwards from now until target_rows reached ---
        print(f"Backfilling {symbol} {interval} → {output_path} (target: {target_rows:,} rows)")
        rows: list[list] = []
        end_ms: int | None = None

        while len(rows) < target_rows:
            kwargs = dict(symbol=symbol, interval=interval, limit=1000)
            if end_ms is not None:
                kwargs["endTime"] = end_ms

            try:
                batch = client.get_klines(**kwargs)
            except Exception as exc:
                print(f"  ERROR: {exc} — retrying in 10s")
                time.sleep(10)
                continue

            if not batch:
                print("  No more data from exchange.")
                break

            rows = batch + rows
            end_ms = int(batch[0][0]) - 1
            print(f"  {len(rows):>7,} rows  oldest={batch[0][0]}")
            time.sleep(0.12)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for row in rows:
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])

        print(f"Done. Saved {len(rows):,} rows → {path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV data from Binance")
    parser.add_argument("--symbol",   default=None, help="Symbol (default: from CRYPTO_PAIR in .env)")
    parser.add_argument("--interval", default=None,
                        help=f"Timeframe: {', '.join(_ML_TIMEFRAMES)}, or 1d. "
                             "Defaults to ML_TIMEFRAME from .env (fallback: 4h)")
    parser.add_argument("--rows",     type=int, default=None, help="Override row count")
    parser.add_argument("--all",      action="store_true",
                        help="Fetch all three ML timeframes (15m, 1h, 4h) plus 1d")
    args = parser.parse_args()

    settings = Settings.from_env()
    client = BinanceRestClient(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        product=settings.binance_product,
        testnet=settings.binance_testnet,
        base_url=settings.binance_base_url,
    )

    sym = (args.symbol or settings.crypto_pair).replace("/", "").upper()

    if args.all:
        intervals = list(_ML_TIMEFRAMES) + ["1d"]
    else:
        interval = args.interval or settings.ml_timeframe
        intervals = [interval, "1d"] if interval != "1d" else ["1d"]

    for interval in intervals:
        rows = args.rows if (args.rows and len(intervals) == 1) else _DEFAULT_ROWS.get(interval, 1_500)
        out = settings.ohlcv_csv_path(sym, interval)
        backfill(client, sym, interval, rows, out)


if __name__ == "__main__":
    main()
