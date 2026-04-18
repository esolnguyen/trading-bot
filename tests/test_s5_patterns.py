"""Acceptance tests for S5 pattern detection and chart rendering."""

from __future__ import annotations

import base64

from src.mcp_servers.shared.domain.analysis import IndicatorSet
from src.mcp_servers.shared.domain.market import OHLCVCandle
from src.legacy.services.analysis import ChartGenerator, PatternAnalyzer


def candle(
    ts: int, open_: float, high: float, low: float, close: float, volume: float = 1000.0
) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=ts, open=open_, high=high, low=low, close=close, volume=volume
    )


def flat_candles(count: int = 60) -> list[OHLCVCandle]:
    return [candle(i * 1000, 100.0, 101.0, 99.0, 100.0) for i in range(count)]


def double_top_candles() -> list[OHLCVCandle]:
    """Two similar highs (≈103/103.5) separated by a deep trough, then decline."""
    values = [
        90,
        93,
        96,
        99,
        101,
        103,
        101,
        98,
        94,
        90,
        90,
        91,
        93,
        97,
        101,
        103.5,
        101,
        97,
        93,
        89,
        85,
        82,
        80,
        78,
        77,
        76,
        75,
        74,
        73,
        72,
    ]
    candles: list[OHLCVCandle] = []
    for index, close in enumerate(values):
        open_ = values[index - 1] if index > 0 else close + 1
        high = max(open_, close) + 1
        low = min(open_, close) - 1
        candles.append(candle(index * 1000, open_, high, low, close))
    return candles


def double_bottom_candles() -> list[OHLCVCandle]:
    values = [
        110,
        108,
        106,
        103,
        100,
        98,
        100,
        103,
        106,
        109,
        107,
        104,
        101,
        99,
        97.5,
        100,
        104,
        108,
        111,
        114,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
    ]
    candles: list[OHLCVCandle] = []
    for index, close in enumerate(values):
        open_ = values[index - 1] if index > 0 else close + 1
        high = max(open_, close) + 1
        low = min(open_, close) - 1
        candles.append(candle(index * 1000, open_, high, low, close))
    return candles


def indicator_fixture() -> IndicatorSet:
    return IndicatorSet(
        rsi_14=42.0,
        macd_line=0.2,
        macd_signal=0.1,
        macd_hist=0.1,
        bb_upper=105.0,
        bb_mid=100.0,
        bb_lower=95.0,
        ema_20=99.0,
        ema_50=98.0,
        volume_sma_20=1000.0,
    )


def test_render_returns_valid_png_base64() -> None:
    rendered = ChartGenerator().render("BTCUSDT", flat_candles(), indicator_fixture())
    payload = base64.b64decode(rendered)

    assert rendered
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_analyze_flat_price_series_returns_no_patterns() -> None:
    result = PatternAnalyzer().analyze("BTCUSDT", flat_candles())

    assert result.patterns == []
    assert result.support is None
    assert result.resistance is None


def test_known_double_bottom_fixture_returns_double_bottom() -> None:
    result = PatternAnalyzer().analyze("BTCUSDT", double_bottom_candles())

    assert "double_bottom" in result.patterns


def test_support_and_resistance_are_none_when_no_levels_found() -> None:
    result = PatternAnalyzer().analyze("ETHUSDT", double_bottom_candles())

    assert result.support is None
    assert result.resistance is None


def test_known_double_top_fixture_returns_double_top() -> None:
    result = PatternAnalyzer().analyze("BTCUSDT", double_top_candles())

    assert "double_top" in result.patterns


def test_double_bottom_does_not_also_fire_double_top() -> None:
    result = PatternAnalyzer().analyze("BTCUSDT", double_bottom_candles())

    assert "double_top" not in result.patterns


def test_double_top_does_not_also_fire_double_bottom() -> None:
    # The double_top fixture also triggers double_bottom internally;
    # the mutual-exclusivity tiebreaker must suppress it.
    result = PatternAnalyzer().analyze("BTCUSDT", double_top_candles())

    assert "double_bottom" not in result.patterns


def test_patterns_never_contain_both_double_bottom_and_top() -> None:
    for fixture in [flat_candles(), double_bottom_candles(), double_top_candles()]:
        result = PatternAnalyzer().analyze("BTCUSDT", fixture)
        both = "double_bottom" in result.patterns and "double_top" in result.patterns
        assert not both, f"Both patterns fired simultaneously on {fixture[:1]}"
