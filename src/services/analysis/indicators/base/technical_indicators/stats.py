"""Mixin: sentiment + statistical indicator methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.services.analysis.indicators.sentiment import fear_and_greed_index_numba
from src.services.analysis.indicators.sentiment.sentiment_indicators import (
    FearGreedConfig,
)
from src.services.analysis.indicators.statistical import (
    apa_adaptive_eot_numba, calculate_eot_numba, entropy_numba, hurst_numba,
    kurtosis_numba, linreg_numba, mad_numba, quantile_numba, skew_numba,
    stdev_numba, variance_numba, zscore_numba,
)


class StatsMixin:
    """Statistical and sentiment indicator methods."""

    # ==================== SENTIMENT ====================

    def fear_and_greed_index(
        self,
        rsi_length: int = 14,
        macd_fast_length: int = 12,
        macd_slow_length: int = 26,
        macd_signal_length: int = 9,
        mfi_length: int = 14,
        window_size: int = 60,
    ) -> np.ndarray:
        config = FearGreedConfig(
            rsi_length=rsi_length,
            macd_fast_length=macd_fast_length,
            macd_slow_length=macd_slow_length,
            macd_signal_length=macd_signal_length,
            mfi_length=mfi_length,
            window_size=window_size,
        )
        return self._base.calculate_indicator(
            fear_and_greed_index_numba,
            self.close, self.high, self.low, self.volume, config,
            required_length=max(
                rsi_length, macd_slow_length + macd_signal_length - 1, mfi_length
            ),
        )

    # ==================== STATISTICAL ====================

    def kurtosis(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            kurtosis_numba, self.close, length, required_length=length,
        )

    def skew(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            skew_numba, self.close, length, required_length=length,
        )

    def stdev(self, length: int = 30, ddof: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            stdev_numba, self.close, length, ddof, required_length=length,
        )

    def variance(self, length: int = 30, ddof: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            variance_numba, self.close, length, ddof, required_length=length,
        )

    def zscore(self, length: int = 30, std: float = 1.0) -> np.ndarray:
        return self._base.calculate_indicator(
            zscore_numba, self.close, length, std, required_length=length,
        )

    def mad(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            mad_numba, self.close, length, required_length=length,
        )

    def quantile(self, length: int = 30, q: float = 0.5) -> np.ndarray:
        return self._base.calculate_indicator(
            quantile_numba, self.close, length, q, required_length=length,
        )

    def entropy(self, length: int = 10, base: float = 2.0) -> np.ndarray:
        return self._base.calculate_indicator(
            entropy_numba, self.close, length, base, required_length=length,
        )

    def hurst(self, max_lag: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            hurst_numba, self.close, max_lag, required_length=max_lag + 2,
        )

    def linreg(self, length: int = 14, r: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            linreg_numba, self.close, length, r, required_length=length,
        )

    def apa_adaptive_eot(
        self,
        q1: float = 0.8, q2: float = 0.4,
        min_len: int = 10, max_len: int = 48, ave_len: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            apa_adaptive_eot_numba, self.close, q1, q2, min_len, max_len, ave_len,
            required_length=max_len,
        )

    def calculate_eot(
        self, length: int = 21, q1: float = 0.8, q2: float = 0.4
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            calculate_eot_numba, self.close, length, q1, q2,
            required_length=length,
        )
