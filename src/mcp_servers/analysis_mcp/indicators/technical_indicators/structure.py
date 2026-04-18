"""Mixin: support/resistance + trend indicator methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.mcp_servers.analysis_mcp.indicators.support_resistance import (
    advanced_support_resistance_numba,
    fibonacci_bollinger_bands_numba,
    fibonacci_pivot_points_numba,
    fibonacci_retracement_numba,
    find_support_resistance_numba,
    floating_levels_numba,
    pivot_points_numba,
    support_resistance_numba,
    support_resistance_numba_advanced,
)
from src.mcp_servers.analysis_mcp.indicators.trend import (
    adx_numba,
    ichimoku_cloud_numba,
    parabolic_sar_numba,
    pfe_numba,
    supertrend_numba,
    td_sequential_numba,
    trix_numba,
    vortex_indicator_numba,
)


class StructureMixin:
    """Support/resistance + trend-structure indicator methods."""

    # ==================== SUPPORT/RESISTANCE ====================

    def support_resistance(self, length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            support_resistance_numba,
            self.high,
            self.low,
            length,
            required_length=length,
        )

    def find_support_resistance(self, window: int = 30) -> Tuple[float, float]:
        support, resistance = self.support_resistance_advanced(length=window)
        return self._base.calculate_indicator(
            find_support_resistance_numba,
            self.close,
            support,
            resistance,
            window,
            required_length=window,
        )

    def support_resistance_advanced(
        self, length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            support_resistance_numba_advanced,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length,
        )

    def advanced_support_resistance(
        self,
        length: int = 25,
        strength_threshold: int = 1,
        persistence: int = 1,
        volume_factor: float = 1.3,
        price_factor: float = 0.004,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            advanced_support_resistance_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            strength_threshold,
            persistence,
            volume_factor,
            price_factor,
            required_length=length,
        )

    def fibonacci_retracement(self, length: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            fibonacci_retracement_numba,
            length,
            self.high,
            self.low,
            required_length=length,
        )

    def fibonacci_bollinger_bands(
        self, length: int = 20, mult: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hlc3 = (self.high + self.low + self.close) / 3
        return self._base.calculate_indicator(
            fibonacci_bollinger_bands_numba,
            hlc3,
            self.volume,
            length,
            mult,
            required_length=length,
        )

    def floating_levels(
        self,
        lookback: int = 20,
        level_up: float = 50.0,
        level_down: float = 50.0,
        length: int = 7,
        multiplier: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            floating_levels_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            lookback,
            level_up,
            level_down,
            required_length=lookback,
        )

    def pivot_points(self) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Standard pivot points and S/R levels: (PP, r1-r4, s1-s4)."""
        return self._base.calculate_indicator(
            pivot_points_numba,
            self.high,
            self.low,
            self.close,
            required_length=1,
        )

    def fibonacci_pivot_points(self) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Fibonacci-ratio pivot points (0.382, 0.618, 1.0): (PP, r1-r3, s1-s3)."""
        return self._base.calculate_indicator(
            fibonacci_pivot_points_numba,
            self.high,
            self.low,
            self.close,
            required_length=1,
        )

    # ==================== TREND ====================

    def adx(self, length: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            adx_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length,
        )

    def supertrend(
        self, length: int = 7, multiplier: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            supertrend_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
        )

    def ichimoku_cloud(
        self,
        conversion_length: int = 9,
        base_length: int = 26,
        lagging_span2_length: int = 52,
        displacement: int = 26,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            ichimoku_cloud_numba,
            self.high,
            self.low,
            conversion_length,
            base_length,
            lagging_span2_length,
            displacement,
            required_length=max(base_length, lagging_span2_length),
        )

    def parabolic_sar(self, step: float = 0.02, max_step: float = 0.2) -> np.ndarray:
        return self._base.calculate_indicator(
            parabolic_sar_numba,
            self.high,
            self.low,
            step,
            max_step,
        )

    def vortex_indicator(self, length: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            vortex_indicator_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length + 1,
        )

    def trix(self, length: int = 18, scalar: float = 100, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            trix_numba,
            self.close,
            length,
            scalar,
            drift,
            required_length=length,
        )

    def pfe(self, n: int = 10, m: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(pfe_numba, self.close, n, m)

    def td_sequential(self, length: int = 9) -> np.ndarray:
        return self._base.calculate_indicator(
            td_sequential_numba,
            self.close,
            length,
            required_length=5,
        )
