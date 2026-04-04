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
        highs = [float(candle.high) for candle in candles]
        lows = [float(candle.low) for candle in candles]
        volumes = [float(candle.volume) for candle in candles]

        rsi_14 = self._rsi(closes, period=14)
        ema_20_series = self._ema_series(closes, period=20)
        ema_50_series = self._ema_series(closes, period=50)
        macd_line_series, macd_signal_series, macd_hist_series = self._macd(closes)
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(closes, period=20, std_dev=2.0)
        volume_sma_20 = self._sma(volumes[-20:])
        vol_ratio = volumes[-1] / volume_sma_20 if volume_sma_20 > 0 else 1.0
        atr = self._atr(highs, lows, closes, period=14)
        adx = self._adx(highs, lows, closes, period=14)
        obv_slope = self._obv_slope(closes, volumes, period=20)
        choppiness = self._choppiness(highs, lows, closes, period=14)
        cci_14 = self._cci(highs, lows, closes, period=14)

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
            atr,
            adx,
            cci_14,
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
            atr=atr,
            adx=adx,
            obv_slope=obv_slope,
            choppiness=choppiness,
            vol_ratio=vol_ratio,
            cci_14=cci_14,
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

    @staticmethod
    def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
        """Compute True Range series (requires at least 2 candles)."""
        trs: list[float] = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return trs

    def _atr(self, highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
        """Wilder-smoothed Average True Range."""
        trs = self._true_ranges(highs, lows, closes)
        if len(trs) < period:
            return 0.0
        # Seed with simple average, then Wilder smoothing
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    def _adx(self, highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
        """Average Directional Index via Wilder smoothing."""
        if len(closes) < period * 2 + 1:
            return 0.0

        plus_dms: list[float] = []
        minus_dms: list[float] = []
        trs = self._true_ranges(highs, lows, closes)

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dms.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dms.append(down_move if down_move > up_move and down_move > 0 else 0.0)

        n = min(len(trs), len(plus_dms), len(minus_dms))
        trs, plus_dms, minus_dms = trs[:n], plus_dms[:n], minus_dms[:n]

        # Seed Wilder sums
        atr_w = sum(trs[:period])
        plus_w = sum(plus_dms[:period])
        minus_w = sum(minus_dms[:period])

        dx_values: list[float] = []
        for i in range(period, n):
            atr_w = atr_w - atr_w / period + trs[i]
            plus_w = plus_w - plus_w / period + plus_dms[i]
            minus_w = minus_w - minus_w / period + minus_dms[i]
            plus_di = 100 * plus_w / atr_w if atr_w > 0 else 0.0
            minus_di = 100 * minus_w / atr_w if atr_w > 0 else 0.0
            di_sum = plus_di + minus_di
            dx_values.append(100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0)

        if not dx_values:
            return 0.0
        # Smooth DX values → ADX
        adx = sum(dx_values[:period]) / period
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period
        return adx

    @staticmethod
    def _obv_slope(closes: list[float], volumes: list[float], period: int) -> float:
        """Linear regression slope of OBV over the last `period` bars, normalised by mean OBV."""
        obv: list[float] = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        window = obv[-period:]
        n = len(window)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(window) / n
        num = sum((i - x_mean) * (window[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0
        # Normalise by absolute mean so the value is scale-independent
        abs_mean = abs(y_mean) if y_mean != 0 else 1.0
        return slope / abs_mean

    @staticmethod
    def _cci(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
        """Commodity Channel Index: (typical_price - SMA_tp) / (0.015 * mean_deviation)."""
        if len(closes) < period:
            return 0.0
        tp = [(h + l + c) / 3.0 for h, l, c in zip(highs[-period:], lows[-period:], closes[-period:])]
        mean_tp = sum(tp) / period
        mad = sum(abs(x - mean_tp) for x in tp) / period
        if mad == 0:
            return 0.0
        return (tp[-1] - mean_tp) / (0.015 * mad)

    def _choppiness(self, highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
        """Choppiness Index: 100 * log10(ΣTR_n / (HH_n - LL_n)) / log10(n).

        Ranges 0–100; values > 61.8 indicate choppy / sideways markets.
        """
        if len(closes) < period + 1:
            return 50.0
        trs = self._true_ranges(highs, lows, closes)
        window_trs = trs[-period:]
        window_highs = highs[-period:]
        window_lows = lows[-period:]
        atr_sum = sum(window_trs)
        hh = max(window_highs)
        ll = min(window_lows)
        price_range = hh - ll
        if price_range <= 0 or atr_sum <= 0:
            return 50.0
        chop = 100.0 * math.log10(atr_sum / price_range) / math.log10(period)
        return max(0.0, min(100.0, chop))
