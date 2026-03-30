"""Chart pattern detection for trading analysis."""

from __future__ import annotations

from src.domain.analysis import PatternResult
from src.domain.market import OHLCVCandle

# Only look at the most recent candles for reversal patterns.
# 200 candles = ~8 days on 1h TF; coincidental price matches are almost
# guaranteed over that window. 60 candles (~2.5 days) is actionable.
_PATTERN_WINDOW = 60
# Minimum candles between the two legs of a double pattern (10h on 1h TF).
_MIN_LEG_SEPARATION = 10
# At least one leg must be within this many candles of the end of the window.
_RECENCY_CANDLES = 20
# Price similarity tolerance for the two legs (1% is tighter than the old 2%).
_LEG_TOLERANCE = 0.01
# The valley/peak between the two legs must be at least this far from the legs.
_NECKLINE_MIN_MOVE = 0.015  # 1.5%


class PatternAnalyzer:
    """Detect simple price-action patterns from recent candles."""

    def analyze(self, symbol: str, candles: list[OHLCVCandle]) -> PatternResult:
        window = candles[-_PATTERN_WINDOW:] if len(candles) > _PATTERN_WINDOW else candles

        double_bottom = self._has_double_bottom(window)
        double_top = self._has_double_top(window)

        # Mutual exclusivity: if both fire, keep only the more recent signal.
        # Compare the index of their right-side leg; higher index = more recent.
        if double_bottom and double_top:
            bottom_recency = self._most_recent_leg_index(window, kind="min")
            top_recency = self._most_recent_leg_index(window, kind="max")
            if bottom_recency >= top_recency:
                double_top = False
            else:
                double_bottom = False

        patterns: list[str] = []
        if double_bottom:
            patterns.append("double_bottom")
        if double_top:
            patterns.append("double_top")

        support = self._find_repeated_level(candles[-50:], kind="support")
        if support is not None:
            patterns.append("support_level")

        resistance = self._find_repeated_level(candles[-50:], kind="resistance")
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

    def _has_double_bottom(self, candles: list[OHLCVCandle]) -> bool:
        """Two similar lows separated by a significant rally, at least one recent."""
        minima = self._local_extrema(candles, kind="min")
        n = len(candles)
        for li, lv in minima:
            for ri, rv in minima:
                if ri - li < _MIN_LEG_SEPARATION:
                    continue
                # Recency: right leg must be near the end of the window
                if ri < n - _RECENCY_CANDLES:
                    continue
                avg = (lv + rv) / 2.0
                if avg == 0:
                    continue
                # Legs must be similar in price
                if abs(lv - rv) / avg > _LEG_TOLERANCE:
                    continue
                # Valley validation: the highest high between the two legs
                # must be meaningfully above the average low (neckline)
                peak_between = max(c.high for c in candles[li:ri + 1])
                if (peak_between - avg) / avg < _NECKLINE_MIN_MOVE:
                    continue
                return True
        return False

    def _has_double_top(self, candles: list[OHLCVCandle]) -> bool:
        """Two similar highs separated by a significant pullback, at least one recent."""
        maxima = self._local_extrema(candles, kind="max")
        n = len(candles)
        for li, lv in maxima:
            for ri, rv in maxima:
                if ri - li < _MIN_LEG_SEPARATION:
                    continue
                # Recency: right leg must be near the end of the window
                if ri < n - _RECENCY_CANDLES:
                    continue
                avg = (lv + rv) / 2.0
                if avg == 0:
                    continue
                # Legs must be similar in price
                if abs(lv - rv) / avg > _LEG_TOLERANCE:
                    continue
                # Neckline validation: the lowest low between the two legs
                # must be meaningfully below the average high
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
