"""Technical indicator calculations for market candles."""

from __future__ import annotations

import math

from src.domain.analysis import IndicatorSet
from src.domain.market import OHLCVCandle


class IndicatorCalculator:
    """Compute a focused set of indicators required by the trading spec."""

    def compute(self, candles: list[OHLCVCandle]) -> IndicatorSet:
        if len(candles) < 50:
            raise ValueError("At least 50 candles are required")

        closes = [float(candle.close) for candle in candles]
        volumes = [float(candle.volume) for candle in candles]

        rsi_14 = self._rsi(closes, period=14)
        ema_20_series = self._ema_series(closes, period=20)
        ema_50_series = self._ema_series(closes, period=50)
        macd_line_series, macd_signal_series, macd_hist_series = self._macd(closes)
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(closes, period=20, std_dev=2.0)
        volume_sma_20 = self._sma(volumes[-20:])

        values = [
            rsi_14,
            ema_20_series[-1],
            ema_50_series[-1],
            macd_line_series[-1],
            macd_signal_series[-1],
            macd_hist_series[-1],
            bb_upper,
            bb_mid,
            bb_lower,
            volume_sma_20,
        ]
        if any(math.isnan(value) or math.isinf(value) for value in values):
            raise ValueError("Indicator calculation produced invalid values")

        return IndicatorSet(
            rsi_14=rsi_14,
            macd_line=macd_line_series[-1],
            macd_signal=macd_signal_series[-1],
            macd_hist=macd_hist_series[-1],
            bb_upper=bb_upper,
            bb_mid=bb_mid,
            bb_lower=bb_lower,
            ema_20=ema_20_series[-1],
            ema_50=ema_50_series[-1],
            volume_sma_20=volume_sma_20,
        )

    @staticmethod
    def _sma(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _ema_series(values: list[float], period: int) -> list[float]:
        multiplier = 2.0 / (period + 1)
        ema_values: list[float] = [sum(values[:period]) / period]
        for value in values[period:]:
            ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    def _rsi(self, closes: list[float], period: int) -> float:
        deltas = [curr - prev for prev, curr in zip(closes[:-1], closes[1:])]
        gains = [max(delta, 0.0) for delta in deltas]
        losses = [max(-delta, 0.0) for delta in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for index in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gains[index]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[index]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _macd(self, closes: list[float]) -> tuple[list[float], list[float], list[float]]:
        ema_fast = self._ema_series(closes, period=12)
        ema_slow = self._ema_series(closes, period=26)

        offset = len(ema_fast) - len(ema_slow)
        macd_line = [fast - slow for fast, slow in zip(ema_fast[offset:], ema_slow)]
        macd_signal = self._ema_series(macd_line, period=9)
        hist_offset = len(macd_line) - len(macd_signal)
        macd_hist = [line - signal for line, signal in zip(macd_line[hist_offset:], macd_signal)]
        macd_line_aligned = macd_line[hist_offset:]

        return macd_line_aligned, macd_signal, macd_hist

    def _bollinger_bands(self, closes: list[float], period: int, std_dev: float) -> tuple[float, float, float]:
        window = closes[-period:]
        mean = self._sma(window)
        variance = sum((value - mean) ** 2 for value in window) / period
        sigma = math.sqrt(variance)
        return mean + std_dev * sigma, mean, mean - std_dev * sigma
