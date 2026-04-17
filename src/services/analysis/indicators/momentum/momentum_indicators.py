"""Backwards-compatible re-export shim.

The numba kernels moved into :mod:`basic`, :mod:`tsi_family`,
:mod:`composite`, and :mod:`ultimate`.  This module keeps the old import
path working.
"""

from .basic import (
    macd_numba,
    momentum_numba,
    roc_numba,
    rsi_numba,
    stochastic_numba,
    williams_r_numba,
)
from .composite import (
    calculate_relative_strength_numba,
    coppock_curve_numba,
    detect_rsi_divergence,
)
from .tsi_family import ppo_numba, rmi_numba, tsi_numba
from .ultimate import UltimateOscillatorConfig, kst_numba, uo_numba

__all__ = [
    "UltimateOscillatorConfig",
    "calculate_relative_strength_numba",
    "coppock_curve_numba",
    "detect_rsi_divergence",
    "kst_numba",
    "macd_numba",
    "momentum_numba",
    "ppo_numba",
    "rmi_numba",
    "roc_numba",
    "rsi_numba",
    "stochastic_numba",
    "tsi_numba",
    "uo_numba",
    "williams_r_numba",
]
