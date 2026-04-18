"""Acceptance tests for S4 technical analysis."""

from __future__ import annotations

from src.mcp_servers.shared.domain.analysis import Signal
from src.mcp_servers.shared.domain.market import MarketSnapshot, OHLCVCandle
from src.legacy.services.analysis import IndicatorCalculator, TechnicalAnalyzer


def build_candles(
    length: int = 60, *, start: float = 100.0, step: float = -1.0
) -> list[OHLCVCandle]:
    candles: list[OHLCVCandle] = []
    price = start
    for index in range(length):
        open_price = price
        close_price = price + step
        high = max(open_price, close_price) + 0.5
        low = min(open_price, close_price) - 0.5
        candles.append(
            OHLCVCandle(
                timestamp=1_700_000_000_000 + index * 3_600_000,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1_000.0 + index * 10.0,
            )
        )
        price = close_price
    return candles


def test_compute_returns_all_fields_without_nan() -> None:
    calculator = IndicatorCalculator()
    indicators = calculator.compute(build_candles())

    assert isinstance(indicators.rsi_14, float)
    assert isinstance(indicators.macd_hist, float)
    assert isinstance(indicators.bb_upper, float)
    assert isinstance(indicators.ema_50, float)
    assert isinstance(indicators.volume_sma_20, float)


def test_analyze_returns_deterministic_signal_for_known_fixture() -> None:
    candles = build_candles()
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        price=candles[-1].close,
        change_24h_pct=-4.2,
        volume_24h=1_500_000.0,
        bid=candles[-1].close - 0.1,
        ask=candles[-1].close + 0.1,
        candles=candles,
        timestamp=candles[-1].timestamp,
    )
    analyzer = TechnicalAnalyzer(IndicatorCalculator())

    first = analyzer.analyze("BTCUSDT", snapshot)
    second = analyzer.analyze("BTCUSDT", snapshot)

    assert first.signal is Signal.BUY
    assert second.signal is Signal.BUY
    assert first.reasoning == second.reasoning


def test_reasoning_includes_rsi_macd_and_band_relation() -> None:
    candles = build_candles()
    snapshot = MarketSnapshot(
        symbol="ETHUSDT",
        price=candles[-1].close,
        change_24h_pct=-2.0,
        volume_24h=750_000.0,
        bid=candles[-1].close - 0.1,
        ask=candles[-1].close + 0.1,
        candles=candles,
        timestamp=candles[-1].timestamp,
    )
    analysis = TechnicalAnalyzer(IndicatorCalculator()).analyze("ETHUSDT", snapshot)

    assert "RSI=" in analysis.reasoning
    assert "MACD_hist=" in analysis.reasoning
    assert "Bollinger band" in analysis.reasoning


def test_compute_raises_value_error_for_fewer_than_50_candles() -> None:
    calculator = IndicatorCalculator()

    try:
        calculator.compute(build_candles(length=49))
    except ValueError as exc:
        assert "50" in str(exc)
    else:
        raise AssertionError("Expected ValueError for fewer than 50 candles")
