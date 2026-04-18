"""E: OHLCV CSV Writer.

Appends the latest closed candle to a local CSV file on every trading
cycle, building up the historical dataset needed for ML training.
Uses stdlib only — no new dependencies.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OHLCVWriter:
    """Append the latest closed candle to a CSV file after each cycle."""

    def __init__(
        self,
        path: str = "data/ohlcv/btcusdt_4h.csv",
        *,
        on_append: Callable[[], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._last_ts: int = 0
        # Optional callback invoked after a new row is written (used to
        # invalidate the HistoricalPercentileScorer in-memory cache).
        self._on_append = on_append
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "open", "high", "low", "close", "volume"])
            logger.info("Created OHLCV CSV: %s", self._path)
        else:
            # Read last timestamp to avoid duplicates
            try:
                with open(self._path, newline="") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    self._last_ts = int(rows[-1]["timestamp"])
            except Exception:  # noqa: BLE001
                pass

    def append(self, candles: list[Any]) -> None:
        """Append candles[-2] (last closed candle) if not already written.

        candles[-1] is the current unclosed candle — always skip it.
        """
        if not candles or len(candles) < 2:
            return
        c = candles[-2]  # last fully closed candle
        ts = int(c.timestamp)
        if ts <= self._last_ts:
            return  # already written

        try:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow([ts, c.open, c.high, c.low, c.close, c.volume])
            self._last_ts = ts
            if self._on_append is not None:
                self._on_append()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OHLCVWriter.append failed: %s", exc)
