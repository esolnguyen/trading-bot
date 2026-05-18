"""Numba-optimized volatility & volume primitives.

Tests shape, NaN-warmup, and a handful of closed-form invariants. The
math itself is exercised end-to-end through TechnicalIndicators + the
catalog tests — this module guards the low-level kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mcp_servers.analysis_mcp.indicators.volatility import (
    atr_numba,
    bollinger_bands_numba,
    chandelier_exit_numba,
    choppiness_index_numba,
    donchian_channels_numba,
    keltner_channels_numba,
)
from src.mcp_servers.analysis_mcp.indicators.volume import (
    ad_line_numba,
    average_quote_volume_numba,
    chaikin_money_flow_numba,
    obv_numba,
    obv_slope_numba,
    rolling_vwap_numba,
    twap_numba,
)


@pytest.fixture
def ohlcv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 100
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + np.abs(rng.normal(0, 0.4, size=n))
    low = close - np.abs(rng.normal(0, 0.4, size=n))
    volume = np.abs(rng.normal(1_000, 200, size=n)).astype(np.float64)
    return high, low, close, volume


class TestATR:
    def test_warmup_is_nan(self, ohlcv) -> None:
        high, low, close, _ = ohlcv
        atr = atr_numba(high, low, close, length=14)
        assert np.all(np.isnan(atr[:13]))
        assert not np.isnan(atr[14])

    def test_atr_is_non_negative(self, ohlcv) -> None:
        high, low, close, _ = ohlcv
        atr = atr_numba(high, low, close, length=14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid >= 0)

    def test_constant_price_atr_zero(self) -> None:
        n = 50
        flat = np.full(n, 100.0)
        atr = atr_numba(flat, flat, flat, length=14)
        valid = atr[~np.isnan(atr)]
        assert np.allclose(valid, 0.0)

    @pytest.mark.parametrize("mode", ["rma", "ema", "sma", "wma"])
    def test_modes_produce_same_shape(self, ohlcv, mode: str) -> None:
        high, low, close, _ = ohlcv
        atr = atr_numba(high, low, close, length=14, mamode=mode)
        assert atr.shape == close.shape


class TestBollingerBands:
    def test_band_ordering(self, ohlcv) -> None:
        _, _, close, _ = ohlcv
        upper, mid, lower = bollinger_bands_numba(close, length=20, num_std_dev=2.0)
        valid = ~np.isnan(upper)
        assert np.all(upper[valid] >= mid[valid])
        assert np.all(mid[valid] >= lower[valid])

    def test_constant_series_bands_collapse(self) -> None:
        flat = np.full(60, 50.0)
        upper, mid, lower = bollinger_bands_numba(flat, length=20, num_std_dev=2.0)
        valid = ~np.isnan(upper)
        assert np.allclose(upper[valid], mid[valid])
        assert np.allclose(mid[valid], lower[valid])


class TestKeltnerChannels:
    def test_band_ordering(self, ohlcv) -> None:
        high, low, close, _ = ohlcv
        upper, mid, lower = keltner_channels_numba(high, low, close, length=20)
        valid = ~np.isnan(upper)
        assert np.all(upper[valid] >= mid[valid])
        assert np.all(mid[valid] >= lower[valid])


class TestDonchianChannels:
    def test_middle_is_average_of_bounds(self, ohlcv) -> None:
        high, low, _, _ = ohlcv
        upper, mid, lower = donchian_channels_numba(high, low, length=20)
        valid = ~np.isnan(upper)
        assert np.allclose(mid[valid], (upper[valid] + lower[valid]) / 2.0)


class TestChandelierExit:
    def test_long_exit_below_recent_high(self, ohlcv) -> None:
        high, low, close, _ = ohlcv
        long_exit, _ = chandelier_exit_numba(high, low, close, length=22, multiplier=3.0)
        valid = ~np.isnan(long_exit)
        # Long exit is HH - mult*ATR → must sit at or below the recent high.
        assert np.all(long_exit[valid] <= np.max(high))


class TestChoppinessIndex:
    def test_within_0_100_range(self, ohlcv) -> None:
        high, low, close, _ = ohlcv
        ci = choppiness_index_numba(high, low, close, length=14)
        valid = ci[~np.isnan(ci)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_short_series_returns_all_nan(self) -> None:
        n = 5
        arr = np.linspace(100, 105, n)
        ci = choppiness_index_numba(arr, arr - 1, arr, length=14)
        assert np.all(np.isnan(ci))


class TestOBV:
    def test_monotone_up_obv_strictly_increasing(self) -> None:
        n = 50
        close = np.arange(100.0, 100.0 + n)
        vol = np.full(n, 10.0)
        obv = obv_numba(close, vol, length=1, initial=1)
        valid = obv[~np.isnan(obv)]
        assert np.all(np.diff(valid) >= 0)

    def test_obv_unchanged_when_close_flat(self) -> None:
        n = 30
        close = np.full(n, 100.0)
        vol = np.full(n, 5.0)
        obv = obv_numba(close, vol, length=1, initial=1)
        valid = obv[~np.isnan(obv)]
        # Flat close → no accumulation past the seed value.
        assert np.allclose(valid, valid[0])

    def test_slope_positive_for_uptrend(self) -> None:
        n = 100
        close = np.arange(100.0, 100.0 + n)
        vol = np.full(n, 10.0)
        obv = obv_numba(close, vol, length=1, initial=1)
        slope = obv_slope_numba(obv, lookback=10)
        valid = slope[50:]  # past the warm-up transient
        assert np.all(np.isfinite(valid))
        assert np.all(valid > 0)

    def test_slope_zero_when_obv_flat(self) -> None:
        n = 30
        close = np.full(n, 100.0)
        vol = np.full(n, 5.0)
        obv = obv_numba(close, vol, length=1, initial=1)
        slope = obv_slope_numba(obv, lookback=10)
        assert np.allclose(slope[20:], 0.0)


class TestCMF:
    def test_within_minus_one_plus_one(self, ohlcv) -> None:
        high, low, close, vol = ohlcv
        cmf = chaikin_money_flow_numba(high, low, close, vol, length=20)
        valid = cmf[~np.isnan(cmf)]
        assert np.all(valid >= -1.0 - 1e-9)
        assert np.all(valid <= 1.0 + 1e-9)


class TestAccumulationDistribution:
    def test_constant_high_low_collapses_to_zero(self) -> None:
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)  # high == low → multiplier guard returns previous
        close = np.full(n, 100.0)
        vol = np.full(n, 10.0)
        ad = ad_line_numba(high, low, close, vol)
        assert np.allclose(ad, 0.0)


class TestVWAP:
    def test_constant_price_vwap_equals_price(self) -> None:
        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        vol = np.full(n, 5.0)
        vwap = rolling_vwap_numba(high, low, close, vol, length=14)
        valid = vwap[~np.isnan(vwap)]
        assert np.allclose(valid, 100.0)


class TestTWAP:
    def test_constant_price_twap_equals_price(self) -> None:
        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        twap = twap_numba(high, low, close, length=14)
        valid = twap[~np.isnan(twap)]
        assert np.allclose(valid, 100.0)


class TestAverageQuoteVolume:
    def test_known_constant_series(self) -> None:
        n = 30
        close = np.full(n, 50.0)
        vol = np.full(n, 4.0)
        out = average_quote_volume_numba(close, vol, window_size=14)
        valid = out[~np.isnan(out)]
        assert np.allclose(valid, 200.0)  # 50 * 4
