"""Basic momentum oscillators: RSI, MACD, Stochastic, ROC, Momentum, Williams %R."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.mcp_servers.analysis_mcp.indicators._numba_compat import njit


@njit(cache=True)
def rsi_numba(close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        diff = float(close[i] - close[i - 1])
        gains[i] = max(0, diff)
        losses[i] = max(0, -diff)

    rsi = np.full(n, np.nan)
    avg_gain = np.sum(gains[1 : length + 1]) / length
    avg_loss = np.sum(losses[1 : length + 1]) / length

    if avg_loss == 0:
        rsi[length] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100 - (100 / (1 + rs))

    for i in range(length + 1, n):
        avg_gain = ((avg_gain * (length - 1)) + gains[i]) / length
        avg_loss = ((avg_loss * (length - 1)) + losses[i]) / length
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@njit(cache=True)
def macd_numba(
    close: np.ndarray,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)

    alpha_fast = 2.0 / (fast_length + 1)
    alpha_slow = 2.0 / (slow_length + 1)
    alpha_signal = 2.0 / (signal_length + 1)

    fast_ema = np.mean(close[:fast_length])
    slow_ema = np.mean(close[:slow_length])
    signal = 0.0

    for i in range(n):
        fast_ema = close[i] * alpha_fast + fast_ema * (1 - alpha_fast)
        slow_ema = close[i] * alpha_slow + slow_ema * (1 - alpha_slow)

        if i >= slow_length - 1:
            macd = fast_ema - slow_ema
            macd_line[i] = macd

            if i == slow_length - 1:
                signal = macd
            elif i > slow_length - 1:
                signal = macd * alpha_signal + signal * (1 - alpha_signal)
                signal_line[i] = signal
                histogram[i] = macd - signal

    return macd_line, signal_line, histogram


@njit(cache=True)
def stochastic_numba(high, low, close, period_k, smooth_k, period_d):
    n = len(close)
    k_values = np.full(n, np.nan)
    d_values = np.full(n, np.nan)

    for i in range(period_k - 1, n):
        start_idx = i - period_k + 1
        end_idx = i + 1

        high_max = high[start_idx]
        low_min = low[start_idx]

        for j in range(start_idx + 1, end_idx):
            val_h = high[j]
            val_l = low[j]
            if val_h > high_max:
                high_max = val_h
            if val_l < low_min:
                low_min = val_l

        if high_max != low_min:
            k_values[i] = 100 * (close[i] - low_min) / (high_max - low_min)

    smoothed_k = np.full(n, np.nan)
    for i in range(period_k + smooth_k - 2, n):
        smoothed_k[i] = np.mean(k_values[i - smooth_k + 1 : i + 1])
        if i >= period_k + smooth_k + period_d - 3:
            d_values[i] = np.mean(smoothed_k[i - period_d + 1 : i + 1])

    return smoothed_k, d_values


@njit(cache=True)
def roc_numba(close, length=1):
    n = len(close)
    roc = np.empty(n, dtype=np.float64)
    roc[:length] = np.nan
    roc[length:] = ((close[length:] / close[:-length]) - 1) * 100
    return roc


@njit(cache=True)
def momentum_numba(close, length=1):
    n = len(close)
    mom = np.full(n, np.nan)
    for i in range(length, n):
        mom[i] = close[i] - close[i - length]
    return mom


@njit(cache=True)
def williams_r_numba(high, low, close, length):
    n = len(close)
    williams_r = np.full(n, np.nan)

    for i in range(length - 1, n):
        start_idx = i - length + 1
        end_idx = i + 1

        highest_high = high[start_idx]
        lowest_low = low[start_idx]

        for j in range(start_idx + 1, end_idx):
            h_val = high[j]
            l_val = low[j]
            if h_val > highest_high:
                highest_high = h_val
            if l_val < lowest_low:
                lowest_low = l_val

        if highest_high != lowest_low:
            williams_r[i] = (
                (highest_high - close[i]) / (highest_high - lowest_low)
            ) * -100

    return williams_r
