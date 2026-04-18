"""Mixin: volatility + volume indicator methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.mcp_servers.analysis_mcp.indicators.volatility import (
    atr_numba,
    bollinger_bands_numba,
    chandelier_exit_numba,
    choppiness_index_numba,
    donchian_channels_numba,
    ebsw_numba,
    keltner_channels_numba,
    vhf_numba,
)
from src.mcp_servers.analysis_mcp.indicators.volume import (
    ad_line_numba,
    average_quote_volume_numba,
    cci_numba,
    chaikin_money_flow_numba,
    eom_numba,
    force_index_numba,
    mfi_numba,
    obv_numba,
    obv_slope_numba,
    pvt_numba,
    rolling_vwap_numba,
    twap_numba,
    volume_profile_numba,
)


class VolatilityVolumeMixin:
    """Volatility bands + volume/flow indicator methods."""

    # ==================== VOLATILITY ====================

    def atr(
        self, length: int = 14, mamode: str = "rma", percent: bool = False
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            atr_numba,
            self.high,
            self.low,
            self.close,
            length,
            mamode,
            percent,
            required_length=length,
        )

    def bollinger_bands(
        self, length: int = 20, num_std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            bollinger_bands_numba,
            self.close,
            length,
            num_std_dev,
            required_length=length,
        )

    def chandelier_exit(
        self, length: int = 22, multiplier: float = 3.0, mamode: str = "rma"
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            chandelier_exit_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            mamode,
        )

    def vhf(self, length: int = 28) -> np.ndarray:
        return self._base.calculate_indicator(
            vhf_numba,
            self.close,
            length,
            required_length=length,
        )

    def ebsw(self, length: int = 40, bars: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            ebsw_numba,
            self.close,
            length,
            bars,
            required_length=length,
        )

    def keltner_channels(
        self, length: int = 20, multiplier: float = 2.0, mamode: str = "ema"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            keltner_channels_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            mamode,
            required_length=length,
        )

    def donchian_channels(
        self, length: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            donchian_channels_numba,
            self.high,
            self.low,
            length,
            required_length=length,
        )

    def choppiness_index(self, length: int = 14) -> np.ndarray:
        """Choppiness index: >61.8 choppy/ranging, <38.2 trending."""
        return self._base.calculate_indicator(
            choppiness_index_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length,
        )

    # ==================== VOLUME ====================

    def cci(self, length: int = 14, c: float = 0.015) -> np.ndarray:
        return self._base.calculate_indicator(
            cci_numba,
            self.high,
            self.low,
            self.close,
            length,
            c,
            required_length=length,
        )

    def mfi(self, length: int = 14, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            mfi_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            drift,
            required_length=length,
        )

    def obv(self, length: int = 14, initial: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            obv_numba,
            self.close,
            self.volume,
            length,
            initial,
            required_length=length,
        )

    def obv_slope(self, length: int = 20, lookback: int = 10) -> np.ndarray:
        """Normalized OBV rate-of-change — accumulation/distribution."""
        obv = self.obv(length=length)
        return self._base.calculate_indicator(
            obv_slope_numba,
            obv,
            lookback,
            required_length=length + lookback,
        )

    def pvt(self, length: int = 14, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            pvt_numba,
            self.close,
            self.volume,
            length,
            drift,
            required_length=length,
        )

    def chaikin_money_flow(self, length: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            chaikin_money_flow_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length,
        )

    def accumulation_distribution_line(self) -> np.ndarray:
        return self._base.calculate_indicator(
            ad_line_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
        )

    def force_index(self, length: int = 13) -> np.ndarray:
        return self._base.calculate_indicator(
            force_index_numba,
            self.close,
            self.volume,
            length,
            required_length=length + 1,
        )

    def eom(
        self, length: int = 14, divisor: int = 100000000, drift: int = 1
    ) -> np.ndarray:
        return self._base.calculate_indicator(
            eom_numba,
            self.high,
            self.low,
            self.volume,
            length,
            divisor,
            drift,
            required_length=length,
        )

    def volume_profile(self, length: int = 48, num_bins: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            volume_profile_numba,
            self.close,
            self.volume,
            length,
            num_bins,
            required_length=length,
        )

    def rolling_vwap(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            rolling_vwap_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length,
        )

    def twap(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            twap_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length,
        )

    def average_quote_volume(self, window_size: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            average_quote_volume_numba,
            self.close,
            self.volume,
            window_size,
            required_length=window_size,
        )
