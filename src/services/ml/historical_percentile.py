"""A1: Historical Percentile Scorer.

Reads the persisted OHLCV CSV and computes percentile ranks for current
indicator values against a rolling 6-month window. Requires numpy only —
no trained model file needed.

CSV data is cached in memory and only re-read when the file modification
time changes (i.e. after OHLCVWriter appends a new row).
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_6M_CANDLES = {
    "15m": 17_280,   # 6 months of 15-minute candles
    "1h":  4_320,    # 6 months of 1-hour candles
    "4h":  1_080,    # 6 months of 4-hour candles
}

# How many candles make up one calendar day per timeframe
_CANDLES_PER_DAY = {
    "15m": 96,
    "1h":  24,
    "4h":  6,
    "1d":  1,
}


class HistoricalPercentileScorer:
    """Compute rolling percentile context for LLM prompt injection."""

    def __init__(self, csv_path: str = "data/ohlcv/btcusdt_4h.csv", timeframe: str = "4h") -> None:
        self._csv_path = Path(csv_path)
        self._timeframe = timeframe
        self._window = _6M_CANDLES.get(timeframe, 1_080)
        self._cpd = _CANDLES_PER_DAY.get(timeframe, 6)  # candles per day
        # In-memory cache — invalidated when the CSV mtime changes.
        self._cache_mtime: float = 0.0
        self._cache_closes: Any = None
        self._cache_highs: Any = None
        self._cache_lows: Any = None
        self._cache_volumes: Any = None

    def invalidate_cache(self) -> None:
        """Force a cache refresh on the next call to score().

        Called by OHLCVWriter after it appends a new row.
        """
        self._cache_mtime = 0.0

    def _load_arrays(self) -> bool:
        """Load (or serve from cache) the numpy arrays for the rolling window.

        Returns True on success, False if data is unavailable or too sparse.
        """
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            logger.debug("numpy not available — percentile scorer disabled")
            return False

        if not self._csv_path.exists():
            logger.debug("OHLCV CSV not found: %s — percentile scorer disabled", self._csv_path)
            return False

        try:
            mtime = os.path.getmtime(self._csv_path)
        except OSError:
            return False

        if mtime == self._cache_mtime and self._cache_closes is not None:
            return True  # cache is fresh

        try:
            rows = self._load_recent_rows(self._window)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read OHLCV CSV: %s", exc)
            return False

        if len(rows) < 500:
            logger.debug("Too few OHLCV rows (%d) for meaningful percentiles", len(rows))
            return False

        self._cache_closes  = np.array([r["close"]  for r in rows], dtype=float)
        self._cache_highs   = np.array([r["high"]   for r in rows], dtype=float)
        self._cache_lows    = np.array([r["low"]    for r in rows], dtype=float)
        self._cache_volumes = np.array([r["volume"] for r in rows], dtype=float)
        self._cache_mtime   = mtime
        logger.debug("HistoricalPercentileScorer cache refreshed (%d rows)", len(rows))
        return True

    def score(self, indicators: Any, current_price: float) -> str | None:
        """Return a formatted percentile summary string or None if data unavailable."""
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            logger.debug("numpy not available — percentile scorer disabled")
            return None

        if not self._load_arrays():
            return None

        closes  = self._cache_closes
        highs   = self._cache_highs
        lows    = self._cache_lows
        volumes = self._cache_volumes

        lines: list[str] = ["## Historical Context (6-month percentiles)"]

        def pct(series: "np.ndarray", value: float) -> int:
            return int(np.sum(series < value) * 100 / len(series))

        # ── RSI ──────────────────────────────────────────────────────────
        rsi_series = self._rolling_rsi(closes, 14)
        if rsi_series is not None and hasattr(indicators, "rsi_14"):
            p = pct(rsi_series, indicators.rsi_14)
            tag = _label(p, high=80, low=20, high_word="overextended", low_word="oversold")
            lines.append(f"- RSI(14): {indicators.rsi_14:.1f} → {p}th pct {tag}".rstrip())

        # ── ADX ──────────────────────────────────────────────────────────
        if hasattr(indicators, "adx"):
            adx_series = self._rolling_adx(closes, highs, lows, 14)
            if adx_series is not None:
                p = pct(adx_series, indicators.adx)
                tag = _label(p, high=80, low=20, high_word="strong trend", low_word="weak trend")
                lines.append(f"- ADX(14): {indicators.adx:.1f} → {p}th pct {tag}".rstrip())

        # ── ATR% ─────────────────────────────────────────────────────────
        if hasattr(indicators, "atr") and indicators.atr > 0:
            atr_pct_series = self._rolling_atr_pct(closes, highs, lows, 14)
            if atr_pct_series is not None:
                current_atr_pct = indicators.atr / current_price * 100
                p = pct(atr_pct_series, current_atr_pct)
                tag = _label(p, high=80, low=20, high_word="high volatility", low_word="low volatility")
                lines.append(f"- ATR%(14): {current_atr_pct:.2f}% → {p}th pct {tag}".rstrip())

        # ── Volume SMA percentile ─────────────────────────────────────────
        if hasattr(indicators, "volume_sma_20") and indicators.volume_sma_20 > 0:
            vol_sma_series = self._rolling_sma(volumes, 20)
            if vol_sma_series is not None:
                p = pct(vol_sma_series, indicators.volume_sma_20)
                tag = _label(p, high=80, low=20, high_word="elevated", low_word="quiet")
                lines.append(f"- Vol SMA(20): {p}th pct {tag}".rstrip())

        # ── Volume expansion: recent 5-candle avg vs 50-candle avg ────────
        if len(volumes) >= 60:
            vol_expansion_series = self._rolling_vol_expansion(volumes, short=5, long=50)
            if vol_expansion_series is not None and len(vol_expansion_series) > 0:
                recent_ratio = float(np.mean(volumes[-5:])) / float(np.mean(volumes[-50:]))
                p = pct(vol_expansion_series, recent_ratio)
                if recent_ratio >= 1.5:
                    tag = "(surging)"
                elif recent_ratio >= 1.2:
                    tag = "(expanding)"
                elif recent_ratio <= 0.6:
                    tag = "(drying up)"
                elif recent_ratio <= 0.8:
                    tag = "(contracting)"
                else:
                    tag = ""
                lines.append(f"- Vol expansion (5/50): {recent_ratio:.2f}× → {p}th pct {tag}".rstrip())

        # ── 7-day price momentum ──────────────────────────────────────────
        lookback_7d = 7 * self._cpd
        if len(closes) > lookback_7d:
            ret7_series = self._rolling_return(closes, lookback_7d)
            if ret7_series is not None and len(ret7_series) > 0:
                current_ret7 = (current_price - closes[-lookback_7d]) / closes[-lookback_7d] * 100
                p = pct(ret7_series, current_ret7)
                tag = _label(p, high=80, low=20, high_word="strong rally", low_word="sharp drop")
                lines.append(f"- 7d momentum: {current_ret7:+.1f}% → {p}th pct {tag}".rstrip())

        # ── 30-day price momentum ─────────────────────────────────────────
        lookback_30d = 30 * self._cpd
        if len(closes) > lookback_30d:
            ret30_series = self._rolling_return(closes, lookback_30d)
            if ret30_series is not None and len(ret30_series) > 0:
                current_ret30 = (current_price - closes[-lookback_30d]) / closes[-lookback_30d] * 100
                p = pct(ret30_series, current_ret30)
                tag = _label(p, high=80, low=20, high_word="strong rally", low_word="sharp drop")
                lines.append(f"- 30d momentum: {current_ret30:+.1f}% → {p}th pct {tag}".rstrip())

        # ── Price position vs 6-month range ──────────────────────────────
        high_6m = float(np.max(highs))
        low_6m  = float(np.min(lows))
        dist_from_high = (current_price - high_6m) / high_6m * 100
        dist_from_low  = (current_price - low_6m)  / low_6m  * 100
        lines.append(f"- Price vs 6M high: {dist_from_high:+.1f}%  vs 6M low: +{dist_from_low:.1f}%")

        if len(lines) <= 1:
            return None
        return "\n".join(lines)

    def _load_recent_rows(self, n: int) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with open(self._csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "close":  float(row["close"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "volume": float(row["volume"]),
                })
        return rows[-n:]

    @staticmethod
    def _rolling_rsi(closes: "Any", period: int) -> "Any | None":
        try:
            import numpy as np  # noqa: PLC0415
            deltas = np.diff(closes)
            gains  = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_g  = np.convolve(gains,  np.ones(period) / period, mode="valid")
            avg_l  = np.convolve(losses, np.ones(period) / period, mode="valid")
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = np.where(avg_l == 0, 100.0, avg_g / avg_l)
            return 100.0 - 100.0 / (1.0 + rs)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rolling_adx(closes: "Any", highs: "Any", lows: "Any", period: int) -> "Any | None":
        """Compute a rolling approximate ADX series using Directional Movement.

        Uses a simple SMA smoothing (not Wilder's EMA) for efficiency, which
        produces values on the same 0-100 scale and is suitable for percentile
        comparison.
        """
        try:
            import numpy as np  # noqa: PLC0415
            n = len(closes)
            if n < period + 2:
                return None

            # True Range
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1]),
                ),
            )

            # Directional movement
            up_move   = highs[1:]  - highs[:-1]
            down_move = lows[:-1]  - lows[1:]
            plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move,   0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            # Smooth with simple SMA
            atr_s    = np.convolve(tr,       np.ones(period) / period, mode="valid")
            plus_s   = np.convolve(plus_dm,  np.ones(period) / period, mode="valid")
            minus_s  = np.convolve(minus_dm, np.ones(period) / period, mode="valid")

            eps = 1e-10
            plus_di  = 100.0 * plus_s  / (atr_s + eps)
            minus_di = 100.0 * minus_s / (atr_s + eps)
            dx       = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + eps)

            # Smooth DX → ADX
            adx = np.convolve(dx, np.ones(period) / period, mode="valid")
            return adx
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rolling_atr_pct(closes: "Any", highs: "Any", lows: "Any", period: int) -> "Any | None":
        try:
            import numpy as np  # noqa: PLC0415
            tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
            atr = np.convolve(tr, np.ones(period) / period, mode="valid")
            atr_pct = atr / closes[period:] * 100
            return atr_pct
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rolling_sma(series: "Any", period: int) -> "Any | None":
        try:
            import numpy as np  # noqa: PLC0415
            return np.convolve(series, np.ones(period) / period, mode="valid")
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rolling_return(closes: "Any", lookback: int) -> "Any | None":
        """Rolling n-candle price return series as percentage."""
        try:
            past   = closes[:-lookback]
            future = closes[lookback:]
            return (future - past) / past * 100
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rolling_vol_expansion(volumes: "Any", short: int, long: int) -> "Any | None":
        """Rolling ratio of short-period vol avg to long-period vol avg."""
        try:
            import numpy as np  # noqa: PLC0415
            sma_short = np.convolve(volumes, np.ones(short) / short, mode="valid")
            sma_long  = np.convolve(volumes, np.ones(long)  / long,  mode="valid")
            n = min(len(sma_short), len(sma_long))
            return sma_short[-n:] / np.where(sma_long[-n:] == 0, 1.0, sma_long[-n:])
        except Exception:  # noqa: BLE001
            return None


def _label(pct: int, high: int, low: int, high_word: str, low_word: str) -> str:
    if pct >= high:
        return f"({high_word})"
    if pct <= low:
        return f"({low_word})"
    return ""
