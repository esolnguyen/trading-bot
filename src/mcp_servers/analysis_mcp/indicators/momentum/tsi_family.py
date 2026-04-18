"""TSI, RMI, and PPO double-smoothed momentum oscillators."""

from __future__ import annotations

import numpy as np

from src.mcp_servers.analysis_mcp.indicators._numba_compat import njit
from src.mcp_servers.analysis_mcp.indicators.overlap import ema_numba


@njit(cache=True)
def tsi_numba(close, long_length, short_length):
    """True Strength Index — double-smoothed momentum, single-pass O(N)."""
    n = len(close)
    tsi = np.full(n, np.nan)

    alpha_long = 2.0 / (long_length + 1)
    alpha_short = 2.0 / (short_length + 1)

    m_sum = 0.0
    abs_m_sum = 0.0

    for i in range(1, long_length + 1):
        if i < n:
            val = close[i] - close[i - 1]
            m_sum += val
            abs_m_sum += abs(val)

    if n <= long_length:
        return tsi

    curr_ema1 = m_sum / long_length
    curr_abs_ema1 = abs_m_sum / long_length

    ema1_sum = curr_ema1
    abs_ema1_sum = curr_abs_ema1

    prev_ema1 = curr_ema1
    prev_abs_ema1 = curr_abs_ema1

    start_ema2_init = long_length + 1
    end_ema2_init = long_length + short_length - 1

    if end_ema2_init >= n:
        return tsi

    for i in range(start_ema2_init, end_ema2_init + 1):
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        ema1_sum += curr_ema1
        abs_ema1_sum += curr_abs_ema1

        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1

    curr_ema2 = ema1_sum / short_length
    curr_abs_ema2 = abs_ema1_sum / short_length

    if curr_abs_ema2 != 0:
        tsi[end_ema2_init] = (curr_ema2 / curr_abs_ema2) * 100.0
    else:
        tsi[end_ema2_init] = 0.0

    prev_ema2 = curr_ema2
    prev_abs_ema2 = curr_abs_ema2

    for i in range(end_ema2_init + 1, n):
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        curr_ema2 = (curr_ema1 - prev_ema2) * alpha_short + prev_ema2
        curr_abs_ema2 = (curr_abs_ema1 - prev_abs_ema2) * alpha_short + prev_abs_ema2

        if curr_abs_ema2 != 0:
            tsi[i] = (curr_ema2 / curr_abs_ema2) * 100.0
        else:
            tsi[i] = tsi[i - 1]

        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1
        prev_ema2 = curr_ema2
        prev_abs_ema2 = curr_abs_ema2

    return tsi


@njit(cache=True)
def rmi_numba(close, length, momentum_length):
    n = len(close)
    rmi = np.full(n, np.nan)

    momentum = np.zeros(n - momentum_length)
    for i in range(len(momentum)):
        momentum[i] = close[i + momentum_length] - close[i]

    up = np.maximum(momentum, 0)
    down = np.maximum(-momentum, 0)

    for i in range(length - 1, len(momentum)):
        avg_up = np.mean(up[i - length + 1 : i + 1])
        avg_down = np.mean(down[i - length + 1 : i + 1])

        if avg_down == 0:
            rmi[i + momentum_length] = 100
        else:
            rs = avg_up / avg_down
            rmi[i + momentum_length] = 100 - (100 / (1 + rs))

    return rmi


@njit(cache=True)
def ppo_numba(close, fast_length, slow_length):
    n = len(close)
    ppo = np.full(n, np.nan)

    fast_ema = ema_numba(close, fast_length)
    slow_ema = ema_numba(close, slow_length)

    for i in range(slow_length - 1, n):
        if slow_ema[i] != 0:
            ppo[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100

    return ppo
