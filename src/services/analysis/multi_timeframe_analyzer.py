"""A2: Multi-Timeframe Alignment Analyzer.

Fetches 1H, 4H, and 1D candles for each symbol and computes a compact
alignment summary for the LLM prompt. Uses existing BinanceFeed and
IndicatorCalculator — no new dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.market import MarketSnapshot, OHLCVCandle
from src.services.analysis.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)

_TIMEFRAMES: list[tuple[str, str]] = [
    ("1h",  "1H"),
    ("4h",  "4H"),
    ("1d",  "1D"),
]
_CANDLE_LIMIT = 200


class MultiTimeframeAnalyzer:
    """Build a multi-timeframe alignment summary for the LLM context."""

    def __init__(self, feed: Any) -> None:
        self._feed = feed
        self._calc = IndicatorCalculator()

    async def build_summary(self, symbol: str, current_tf_signal: str) -> str | None:
        """Return a formatted multi-TF summary string or None on failure."""
        lines: list[str] = [f"## Timeframe Alignment ({symbol.replace('USDT', '')})"]
        lines.append(f"- Current TF: {current_tf_signal}")

        tf_results: list[tuple[str, str]] = []
        for tf, label in _TIMEFRAMES:
            result = await self._analyze_tf(symbol, tf, label)
            if result:
                tf_results.append((label, result))

        if not tf_results:
            return None

        for label, summary in tf_results:
            lines.append(f"- {label}: {summary}")

        # Alignment score
        bullish_count = sum(
            1 for _, s in tf_results
            if any(w in s.upper() for w in ("BULLISH", "STRONG_BUY", "BUY"))
        )
        bearish_count = sum(
            1 for _, s in tf_results
            if any(w in s.upper() for w in ("BEARISH", "STRONG_SELL", "SELL"))
        )
        total = len(tf_results)
        if bullish_count > bearish_count:
            alignment = f"{bullish_count}/{total} bullish"
            note = "" if bullish_count == total else " — partial alignment, proceed with caution"
        elif bearish_count > bullish_count:
            alignment = f"{bearish_count}/{total} bearish"
            note = "" if bearish_count == total else " — partial alignment, proceed with caution"
        else:
            alignment = "mixed/neutral"
            note = " — conflicting signals, prefer HOLD"
        lines.append(f"→ Alignment: {alignment}{note}")

        return "\n".join(lines)

    async def _analyze_tf(self, symbol: str, timeframe: str, label: str) -> str | None:
        try:
            candles: list[OHLCVCandle] = await self._feed.get_ohlcv(
                symbol, timeframe=timeframe, limit=_CANDLE_LIMIT
            )
            if len(candles) < 50:
                return None
            ind = self._calc.compute(candles)

            signal = _derive_signal(ind)
            details = (
                f"RSI {ind.rsi_14:.0f}, "
                f"MACD {'▲' if ind.macd_hist > 0 else '▼'} ({ind.macd_hist:+.2f}), "
                f"ADX {ind.adx:.0f}"
            )
            return f"{signal} — {details}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("MultiTF %s/%s failed: %s", symbol, timeframe, exc)
            return None


def _derive_signal(ind: Any) -> str:
    """Simple rule-based signal from indicators — mirrors TechnicalAnalyzer logic."""
    score = 0
    if ind.rsi_14 > 55:
        score += 1
    elif ind.rsi_14 < 45:
        score -= 1
    if ind.rsi_14 > 70:
        score -= 1  # overbought penalty
    elif ind.rsi_14 < 30:
        score += 1  # oversold bonus

    if ind.macd_hist > 0:
        score += 1
    else:
        score -= 1

    if ind.ema_20 > ind.ema_50:
        score += 1
    else:
        score -= 1

    if ind.adx > 25:
        if score > 0:
            score += 1
        else:
            score -= 1

    if score >= 3:
        return "STRONG_BUY"
    if score >= 1:
        return "BULLISH"
    if score <= -3:
        return "STRONG_SELL"
    if score <= -1:
        return "BEARISH"
    return "NEUTRAL"
