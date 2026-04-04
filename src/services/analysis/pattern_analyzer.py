"""Chart pattern detection for trading analysis."""

from __future__ import annotations

from src.domain.analysis import PatternResult
from src.domain.market import OHLCVCandle

# Candle window scaled by timeframe so the lookback covers a consistent
# wall-clock duration regardless of candle size.
# Target: ~1.5 days of price action for reversal patterns.
_WINDOW_BY_TF: dict[str, int] = {
    "1m":  2160,  # 1.5 days
    "5m":   432,  # 1.5 days
    "15m":  144,  # 1.5 days
    "30m":   72,  # 1.5 days
    "1h":    60,  # 2.5 days  (original default)
    "2h":    30,  # 2.5 days
    "4h":    20,  # 3.3 days
    "6h":    16,  # 4 days
    "8h":    12,  # 4 days
    "12h":    8,  # 4 days
    "1d":    30,  # 1 month
}
_DEFAULT_WINDOW = 60

# Leg separation and recency also scale with window size so the proportions
# stay meaningful (leg_sep ≈ window/6, recency ≈ window/3).
_LEG_TOLERANCE      = 0.01   # legs must match within 1%
_NECKLINE_MIN_MOVE  = 0.015  # valley/peak must be ≥1.5% from the legs


class PatternAnalyzer:
    """Detect simple price-action patterns from recent candles."""

    def analyze(
        self,
        symbol: str,
        candles: list[OHLCVCandle],
        timeframe: str = "1h",
    ) -> PatternResult:
        window_size    = _WINDOW_BY_TF.get(timeframe, _DEFAULT_WINDOW)
        leg_sep        = max(5, window_size // 6)   # minimum candles between legs
        recency        = max(5, window_size // 3)   # one leg must be in last N candles

        window = candles[-window_size:] if len(candles) > window_size else candles

        double_bottom = self._has_double_bottom(window, leg_sep, recency)
        double_top    = self._has_double_top(window, leg_sep, recency)

        # Mutual exclusivity: keep the more recent signal.
        if double_bottom and double_top:
            bottom_recency = self._most_recent_leg_index(window, kind="min")
            top_recency    = self._most_recent_leg_index(window, kind="max")
            if bottom_recency >= top_recency:
                double_top = False
            else:
                double_bottom = False

        patterns: list[str] = []
        if double_bottom:
            patterns.append("double_bottom")
        if double_top:
            patterns.append("double_top")

        # Support/resistance: scan last 50 candles (or full window for short TFs)
        sr_window = min(50, window_size)
        support    = self._find_repeated_level(candles[-sr_window:], kind="support")
        if support is not None:
            patterns.append("support_level")

        resistance = self._find_repeated_level(candles[-sr_window:], kind="resistance")
        if resistance is not None:
            patterns.append("resistance_level")

        engulfing = self._engulfing_pattern(candles)
        if engulfing is not None:
            patterns.append(engulfing)

        return PatternResult(
            symbol=symbol,
            patterns=patterns,
            support=support,
            resistance=resistance,
            chart_png_b64="",
        )

    # ------------------------------------------------------------------
    # Double bottom / top
    # ------------------------------------------------------------------

    def _has_double_bottom(self, candles: list[OHLCVCandle], leg_sep: int, recency: int) -> bool:
        """Two similar lows separated by a significant rally, at least one recent."""
        minima = self._local_extrema(candles, kind="min")
        n = len(candles)
        for li, lv in minima:
            for ri, rv in minima:
                if ri - li < leg_sep:
                    continue
                if ri < n - recency:
                    continue
                avg = (lv + rv) / 2.0
                if avg == 0:
                    continue
                if abs(lv - rv) / avg > _LEG_TOLERANCE:
                    continue
                peak_between = max(c.high for c in candles[li:ri + 1])
                if (peak_between - avg) / avg < _NECKLINE_MIN_MOVE:
                    continue
                return True
        return False

    def _has_double_top(self, candles: list[OHLCVCandle], leg_sep: int, recency: int) -> bool:
        """Two similar highs separated by a significant pullback, at least one recent."""
        maxima = self._local_extrema(candles, kind="max")
        n = len(candles)
        for li, lv in maxima:
            for ri, rv in maxima:
                if ri - li < leg_sep:
                    continue
                if ri < n - recency:
                    continue
                avg = (lv + rv) / 2.0
                if avg == 0:
                    continue
                if abs(lv - rv) / avg > _LEG_TOLERANCE:
                    continue
                trough_between = min(c.low for c in candles[li:ri + 1])
                if (avg - trough_between) / avg < _NECKLINE_MIN_MOVE:
                    continue
                return True
        return False

    def _most_recent_leg_index(self, candles: list[OHLCVCandle], kind: str) -> int:
        """Return the index of the most recent local extremum."""
        extrema = self._local_extrema(candles, kind=kind)
        return extrema[-1][0] if extrema else 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _local_extrema(self, candles: list[OHLCVCandle], kind: str) -> list[tuple[int, float]]:
        extrema: list[tuple[int, float]] = []
        for index in range(1, len(candles) - 1):
            prev_candle = candles[index - 1]
            candle = candles[index]
            next_candle = candles[index + 1]
            if kind == "min":
                if (
                    candle.low <= prev_candle.low
                    and candle.low <= next_candle.low
                    and (candle.low < prev_candle.low or candle.low < next_candle.low)
                ):
                    extrema.append((index, candle.low))
            if kind == "max":
                if (
                    candle.high >= prev_candle.high
                    and candle.high >= next_candle.high
                    and (candle.high > prev_candle.high or candle.high > next_candle.high)
                ):
                    extrema.append((index, candle.high))
        return extrema

    def _find_repeated_level(self, candles: list[OHLCVCandle], *, kind: str) -> float | None:
        if len(candles) < 3:
            return None

        extrema = self._local_extrema(candles, kind="min" if kind == "support" else "max")
        if len(extrema) < 3:
            return None

        touches: list[float] = []
        for _index, level in extrema:
            if not any(abs(level - existing) <= existing * 0.01 for existing in touches):
                touches.append(level)

        for candidate in touches:
            tolerance = candidate * 0.01
            count = 0
            last_match_index: int | None = None
            for index, level in extrema:
                if abs(level - candidate) > tolerance:
                    continue
                if last_match_index is None or index - last_match_index >= 3:
                    count += 1
                    last_match_index = index
            if count >= 3:
                return candidate
        return None

    def _engulfing_pattern(self, candles: list[OHLCVCandle]) -> str | None:
        if len(candles) < 2:
            return None

        previous = candles[-2]
        current = candles[-1]
        prev_body_low = min(previous.open, previous.close)
        prev_body_high = max(previous.open, previous.close)
        curr_body_low = min(current.open, current.close)
        curr_body_high = max(current.open, current.close)

        if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
            if current.close > current.open:
                return "bullish_engulfing"
            if current.close < current.open:
                return "bearish_engulfing"
        return None
