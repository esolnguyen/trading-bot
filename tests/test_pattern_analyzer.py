"""PatternAnalyzer: double-top/bottom, engulfing, S/R detection."""

from __future__ import annotations

from src.mcp_servers.shared.domain.market import OHLCVCandle
from src.mcp_servers.shared.services import PatternAnalyzer


def _candle(
    i: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1.0,
) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=i * 60_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _flat(n: int, price: float = 100.0) -> list[OHLCVCandle]:
    return [_candle(i, price, price + 0.1, price - 0.1, price) for i in range(n)]


def test_flat_market_no_patterns() -> None:
    analyzer = PatternAnalyzer()
    result = analyzer.analyze("BTCUSDT", _flat(60))
    assert result.symbol == "BTCUSDT"
    assert result.patterns == []
    assert result.support is None
    assert result.resistance is None


def _arc(start: float, peak: float, n: int) -> list[float]:
    """Build a smooth single-arc path of length n from start → peak → start."""
    import math

    return [
        start + (peak - start) * math.sin(math.pi * i / (n - 1)) for i in range(n)
    ]


def _valley(start: float, trough: float, n: int) -> list[float]:
    """Single-valley path of length n from start → trough → start."""
    import math

    return [
        start - (start - trough) * math.sin(math.pi * i / (n - 1)) for i in range(n)
    ]


def _path_to_candles(prices: list[float]) -> list[OHLCVCandle]:
    """Turn a list of closes into 1-tick OHLCV candles with tiny wicks."""
    return [
        _candle(i, prices[i], prices[i] + 0.05, prices[i] - 0.05, prices[i])
        for i in range(len(prices))
    ]


def test_double_bottom_detected() -> None:
    # 1h timeframe → window=60, recency=20. Two valleys @ ~99 separated
    # by a single arc rally → only one local-max in between, no spurious
    # double_top. Second valley sits in the final 20 candles.
    analyzer = PatternAnalyzer()
    prices: list[float] = []
    prices.extend(_valley(start=104.0, trough=99.0, n=10))   # first bottom @ ~5
    prices.extend(_arc(start=99.0, peak=105.0, n=30))        # neckline rally
    prices.extend(_valley(start=105.0, trough=99.05, n=15))  # second bottom @ ~52
    prices.extend([101.0] * 5)
    candles = _path_to_candles(prices)
    result = analyzer.analyze("BTCUSDT", candles, timeframe="1h")
    assert "double_bottom" in result.patterns


def test_double_top_detected() -> None:
    analyzer = PatternAnalyzer()
    prices: list[float] = []
    prices.extend(_arc(start=95.0, peak=100.0, n=10))        # first top @ ~5
    prices.extend(_valley(start=100.0, trough=95.0, n=30))   # neckline pullback
    prices.extend(_arc(start=95.0, peak=100.05, n=15))       # second top @ ~52
    prices.extend([97.0] * 5)
    candles = _path_to_candles(prices)
    result = analyzer.analyze("BTCUSDT", candles, timeframe="1h")
    assert "double_top" in result.patterns


def test_bullish_engulfing_pattern() -> None:
    analyzer = PatternAnalyzer()
    candles = _flat(58)
    # Prior candle: bearish small body
    candles.append(_candle(58, 101.0, 101.2, 100.8, 100.9))
    # Engulfing candle: bullish, body completely covers prior body
    candles.append(_candle(59, 100.7, 101.6, 100.6, 101.5))
    result = analyzer.analyze("BTCUSDT", candles)
    assert "bullish_engulfing" in result.patterns


def test_bearish_engulfing_pattern() -> None:
    analyzer = PatternAnalyzer()
    candles = _flat(58)
    candles.append(_candle(58, 100.9, 101.1, 100.8, 101.0))  # prior bullish
    candles.append(_candle(59, 101.1, 101.2, 100.5, 100.6))  # bearish engulf
    result = analyzer.analyze("BTCUSDT", candles)
    assert "bearish_engulfing" in result.patterns


def test_resistance_level_detected_from_repeated_highs() -> None:
    analyzer = PatternAnalyzer()
    # Three pivots at the same high (~110) with valid spacing.
    candles: list[OHLCVCandle] = []
    for i in range(40):
        # Inject pivots at i=8, 20, 32 — local maxes at 110.
        if i in {8, 20, 32}:
            candles.append(_candle(i, 105.0, 110.0, 104.0, 106.0))
        else:
            candles.append(_candle(i, 100.0, 101.0, 99.0, 100.5))
    result = analyzer.analyze("BTCUSDT", candles)
    assert "resistance_level" in result.patterns
    assert result.resistance is not None
    assert abs(result.resistance - 110.0) < 0.01


def test_double_top_and_bottom_mutually_exclusive() -> None:
    # When both can be detected, the more recent leg wins.
    analyzer = PatternAnalyzer()
    candles: list[OHLCVCandle] = []
    # Old double bottom early in series.
    candles.append(_candle(0, 100.5, 101.0, 99.9, 100.5))
    candles.append(_candle(1, 100.5, 101.0, 100.0, 101.0))
    for i in range(2, 12):
        candles.append(_candle(i, 102.0, 104.5, 101.5, 104.0))
    candles.append(_candle(12, 100.5, 101.0, 99.9, 100.5))
    # Now a fresh double top very late in the window.
    for i in range(13, 35):
        candles.append(_candle(i, 102.0, 102.5, 101.5, 102.0))
    candles.append(_candle(35, 109.5, 110.0, 109.0, 109.8))
    for i in range(36, 55):
        candles.append(_candle(i, 105.0, 106.0, 104.0, 104.5))
    candles.append(_candle(55, 109.5, 110.0, 109.0, 109.8))

    result = analyzer.analyze("BTCUSDT", candles, timeframe="1h")
    assert not ("double_top" in result.patterns and "double_bottom" in result.patterns)


def test_engulfing_requires_two_candles() -> None:
    analyzer = PatternAnalyzer()
    result = analyzer.analyze("BTCUSDT", [_candle(0, 100, 101, 99, 100.5)])
    assert "bullish_engulfing" not in result.patterns
    assert "bearish_engulfing" not in result.patterns
