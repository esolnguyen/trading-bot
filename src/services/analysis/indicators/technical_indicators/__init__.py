"""Technical Indicators — direct method access via category mixins.

All indicator methods are exposed directly on :class:`TechnicalIndicators`
instead of through category sub-objects.  Implementation lives in mixin
modules to keep each file small:

  * :mod:`.oscillators` — momentum, overlap, price
  * :mod:`.stats`       — sentiment, statistical
  * :mod:`.structure`   — support/resistance, trend
  * :mod:`.volume`      — volatility, volume
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..indicator_base import IndicatorBase
from .oscillators import OscillatorsMixin
from .stats import StatsMixin
from .structure import StructureMixin
from .volume import VolatilityVolumeMixin


class TechnicalIndicators(
    OscillatorsMixin,
    StatsMixin,
    StructureMixin,
    VolatilityVolumeMixin,
):
    """Technical indicators class with direct method access.

    Usage:
        ti = TechnicalIndicators()
        ti.get_data(ohlcv_data)
        rsi_values = ti.rsi(length=14)
    """
    # pylint: disable=too-many-public-methods

    def __init__(
        self, measure_time: bool = False, save_to_csv: bool = False
    ) -> None:
        self._base = IndicatorBase(
            measure_time=measure_time, save_to_csv=save_to_csv
        )

    # ==================== PROPERTIES ====================

    @property
    def open(self) -> np.ndarray:
        return self._base.open

    @property
    def high(self) -> np.ndarray:
        return self._base.high

    @property
    def low(self) -> np.ndarray:
        return self._base.low

    @property
    def close(self) -> np.ndarray:
        return self._base.close

    @property
    def volume(self) -> np.ndarray:
        return self._base.volume

    def get_data(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]],
    ) -> None:
        self._base.get_data(data)
