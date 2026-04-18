"""Mixin: momentum oscillators + overlap (moving averages) + price transforms."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.mcp_servers.analysis_mcp.indicators.momentum import (
    calculate_relative_strength_numba,
    coppock_curve_numba,
    detect_rsi_divergence,
    kst_numba,
    macd_numba,
    momentum_numba,
    ppo_numba,
    rmi_numba,
    roc_numba,
    rsi_numba,
    stochastic_numba,
    tsi_numba,
    uo_numba,
    williams_r_numba,
)
from src.mcp_servers.analysis_mcp.indicators.overlap import (
    ema_numba,
    ewma_numba,
    sma_numba,
)
from src.mcp_servers.analysis_mcp.indicators.price import (
    log_return_numba,
    pdist_numba,
    percent_return_numba,
)


class OscillatorsMixin:
    """Momentum, overlap and price-transform indicator methods."""

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            rsi_numba,
            self.close,
            length,
            required_length=length,
        )

    def macd(
        self,
        fast_length: int = 12,
        slow_length: int = 26,
        signal_length: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            macd_numba,
            self.close,
            fast_length,
            slow_length,
            signal_length,
            required_length=slow_length,
        )

    def stochastic(
        self,
        period_k: int = 5,
        smooth_k: int = 3,
        period_d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            stochastic_numba,
            self.high,
            self.low,
            self.close,
            period_k,
            smooth_k,
            period_d,
            required_length=3,
        )

    def roc(self, length: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            roc_numba,
            self.close,
            length,
            required_length=length + 1,
        )

    def momentum(self, length: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            momentum_numba,
            self.close,
            length,
            required_length=length,
        )

    def williams_r(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            williams_r_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length,
        )

    def tsi(self, long_length: int = 25, short_length: int = 13) -> np.ndarray:
        return self._base.calculate_indicator(
            tsi_numba,
            self.close,
            long_length,
            short_length,
            required_length=long_length + short_length,
        )

    def rmi(self, length: int = 14, momentum_length: int = 5) -> np.ndarray:
        return self._base.calculate_indicator(
            rmi_numba,
            self.close,
            length,
            momentum_length,
            required_length=length + momentum_length,
        )

    def ppo(self, fast_length: int = 12, slow_length: int = 26) -> np.ndarray:
        return self._base.calculate_indicator(
            ppo_numba,
            self.close,
            fast_length,
            slow_length,
            required_length=max(fast_length, slow_length),
        )

    def coppock_curve(
        self, wl1: int = 14, wl2: int = 11, wma_length: int = 10
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            coppock_curve_numba,
            self.close,
            wl1,
            wl2,
            wma_length,
        )

    def detect_rsi_divergence(
        self, rsi_values: np.ndarray, length: int = 14
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            detect_rsi_divergence,
            self.close,
            rsi_values,
            length,
            required_length=length,
        )

    def relative_strength_index(
        self, benchmark_close: np.ndarray, window: int = 14
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            calculate_relative_strength_numba,
            self.close,
            benchmark_close,
            window,
            required_length=window,
        )

    def kst(
        self,
        roc1_length: int = 5,
        roc2_length: int = 10,
        roc3_length: int = 15,
        roc4_length: int = 20,
        sma1_length: int = 3,
        sma2_length: int = 5,
        sma3_length: int = 7,
        sma4_length: int = 9,
    ) -> np.ndarray:
        """Know Sure Thing (KST) oscillator — NaN-aware implementation."""
        return self._base.calculate_indicator(
            kst_numba,
            self.close,
            roc1_length,
            roc2_length,
            roc3_length,
            roc4_length,
            sma1_length,
            sma2_length,
            sma3_length,
            sma4_length,
            required_length=roc4_length + sma4_length,
        )

    def uo(
        self,
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
        fast_w: float = 4.0,
        medium_w: float = 2.0,
        slow_w: float = 1.0,
        drift: int = 1,
    ) -> np.ndarray:
        config = {
            "fast": fast,
            "medium": medium,
            "slow": slow,
            "fast_w": fast_w,
            "medium_w": medium_w,
            "slow_w": slow_w,
            "drift": drift,
        }
        return self._base.calculate_indicator(
            uo_numba,
            self.high,
            self.low,
            self.close,
            config,
            required_length=slow,
        )

    # ==================== OVERLAP INDICATORS ====================

    def ema(self, data_series: np.ndarray, length: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            ema_numba,
            data_series,
            length,
            required_length=length,
        )

    def sma(self, data_series: np.ndarray, length: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            sma_numba,
            data_series,
            length,
            required_length=length,
        )

    def ewma(self, span: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(ewma_numba, self.close, span)

    # ==================== PRICE TRANSFORM INDICATORS ====================

    def log_return(self, length: int = 1, cumulative: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            log_return_numba,
            self.close,
            length,
            cumulative,
            required_length=length,
        )

    def percent_return(self, length: int = 1, cumulative: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            percent_return_numba,
            self.close,
            length,
            cumulative,
            required_length=length,
        )

    def pdist(self, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            pdist_numba,
            self.open,
            self.high,
            self.low,
            self.close,
            drift,
            required_length=1,
        )
