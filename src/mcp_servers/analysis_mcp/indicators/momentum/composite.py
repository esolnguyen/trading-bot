"""Composite momentum indicators: Coppock curve, RSI divergence, relative strength."""

from __future__ import annotations

import numpy as np

from src.mcp_servers.analysis_mcp.indicators._numba_compat import njit
from src.mcp_servers.analysis_mcp.indicators.overlap import ema_numba

from .basic import roc_numba


@njit(cache=True)
def coppock_curve_numba(close, wl1=14, wl2=11, wma_length=10):
    roc_long = roc_numba(close, wl1)
    roc_short = roc_numba(close, wl2)
    coppock_arr = roc_long + roc_short
    # ema_numba handles NaNs correctly, avoiding lookahead bias from np.roll
    return ema_numba(coppock_arr, wma_length)


@njit(cache=True)
def detect_rsi_divergence(close_prices, rsi_values, length=14):
    divergence = np.zeros_like(close_prices)
    n = len(close_prices)
    for i in range(length, n):
        price_diff = close_prices[i] - close_prices[i - length]
        rsi_diff = rsi_values[i] - rsi_values[i - length]
        if price_diff < 0 and rsi_diff > 0:
            divergence[i] = 1
        elif price_diff > 0 and rsi_diff < 0:
            divergence[i] = -1
        else:
            divergence[i] = 0
    return divergence


@njit(cache=True)
def calculate_relative_strength_numba(pair_close, benchmark_close, window=14):
    n = len(pair_close)
    rs_array = np.zeros(n)

    for i in range(window, n):
        if (
            np.isnan(pair_close[i])
            or np.isnan(benchmark_close[i])
            or benchmark_close[i] == 0
        ):
            rs_array[i] = 0.0
            continue

        pair_return = np.log(pair_close[i] / pair_close[i - window])
        benchmark_return = np.log(benchmark_close[i] / benchmark_close[i - window])
        rs_value = pair_return - benchmark_return

        # Cap at ±0.5 to prevent extreme scores
        rs_array[i] = min(max(float(rs_value), -0.5), 0.5)

    return rs_array
