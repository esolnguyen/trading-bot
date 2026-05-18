"""Indicator math on synthetic OHLCV series.

Asserts on shape, invariants, and a handful of closed-form values rather
than reimplementing the math — that would just tautologically re-test
the calculator.
"""

from __future__ import annotations

import math

import pytest

from src.mcp_servers.shared.domain.market import OHLCVCandle
from src.mcp_servers.shared.services import IndicatorCalculator


def _candles_constant(n: int, price: float = 100.0, volume: float = 1.0) -> list[OHLCVCandle]:
    return [
        OHLCVCandle(
            timestamp=i * 60_000,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
        )
        for i in range(n)
    ]


def _candles_trend(n: int, start: float = 100.0, step: float = 1.0) -> list[OHLCVCandle]:
    candles: list[OHLCVCandle] = []
    for i in range(n):
        close = start + step * i
        candles.append(
            OHLCVCandle(
                timestamp=i * 60_000,
                open=close - step / 2,
                high=close + 0.5,
                low=close - step - 0.5,
                close=close,
                volume=1.0 + (i % 5),
            )
        )
    return candles


def test_rejects_short_history() -> None:
    calc = IndicatorCalculator()
    with pytest.raises(ValueError, match="50 candles"):
        calc.compute(_candles_constant(49))


def test_constant_series_rsi_clamps_to_100() -> None:
    # No down moves → avg_loss == 0 → divide-by-zero guard returns 100.
    calc = IndicatorCalculator()
    result = calc.compute(_candles_constant(100))
    assert result.rsi_14 == 100.0


def test_uptrend_pushes_rsi_above_neutral() -> None:
    calc = IndicatorCalculator()
    result = calc.compute(_candles_trend(120, step=1.0))
    assert result.rsi_14 > 50.0


def test_indicator_set_has_no_nan_or_inf() -> None:
    calc = IndicatorCalculator()
    result = calc.compute(_candles_trend(120))
    for value in (
        result.rsi_14,
        result.macd_line,
        result.macd_signal,
        result.macd_hist,
        result.bb_upper,
        result.bb_mid,
        result.bb_lower,
        result.ema_20,
        result.ema_50,
        result.atr,
        result.adx,
        result.choppiness,
        result.cci_14,
        result.volume_sma_20,
    ):
        assert not math.isnan(value)
        assert not math.isinf(value)


def test_bollinger_band_ordering() -> None:
    calc = IndicatorCalculator()
    result = calc.compute(_candles_trend(120))
    assert result.bb_lower <= result.bb_mid <= result.bb_upper


def test_constant_series_bollinger_collapses_to_mid() -> None:
    # Zero variance → sigma=0 → all three bands equal.
    calc = IndicatorCalculator()
    result = calc.compute(_candles_constant(100, price=42.0))
    assert result.bb_upper == result.bb_mid == result.bb_lower == 42.0


def test_constant_series_choppiness_neutral() -> None:
    # No price range → guard returns the neutral 50.0.
    calc = IndicatorCalculator()
    result = calc.compute(_candles_constant(100))
    assert result.choppiness == 50.0


def test_trending_series_choppiness_below_threshold() -> None:
    # A strictly monotonic move should register as trending (< 61.8).
    calc = IndicatorCalculator()
    result = calc.compute(_candles_trend(120, step=2.0))
    assert result.choppiness < 61.8


def test_volume_ratio_picks_up_spike() -> None:
    calc = IndicatorCalculator()
    candles = _candles_trend(120)
    # Replace the last bar with a 10× volume spike.
    last = candles[-1]
    candles[-1] = OHLCVCandle(
        timestamp=last.timestamp,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume * 50,
    )
    result = calc.compute(candles)
    assert result.vol_ratio > 5.0


def test_uptrend_makes_ema20_below_close() -> None:
    # EMA lags a fresh uptrend, so EMA(20) sits below the latest close.
    calc = IndicatorCalculator()
    candles = _candles_trend(120, step=1.0)
    result = calc.compute(candles)
    assert result.ema_20 < candles[-1].close


def test_uptrend_adx_signals_directional_strength() -> None:
    calc = IndicatorCalculator()
    result = calc.compute(_candles_trend(120, step=2.0))
    # A clean linear uptrend should sit well above the "weak" threshold (25).
    assert result.adx > 25.0
