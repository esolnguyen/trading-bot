"""Catalog of indicators exposed by the Analysis MCP category tools.

Each entry maps an MCP-facing name → (facade method, output labels, defaults).
``labels`` is the list of series names emitted by the facade method:

* single-series returns (e.g. ``rsi``): one label, one ndarray.
* tuple returns (e.g. ``macd`` → (line, signal, hist); ``bollinger`` →
  (upper, mid, lower); ``ichimoku`` → 5 series): one label per element.

Handlers call :func:`run_indicator` and map the result onto pydantic
response models — no direct dispatch lives in the handlers themselves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.mcp_servers.analysis_mcp.indicators import TechnicalIndicators


# (facade_method, output_labels, default_params)
_CATALOG: dict[str, dict[str, tuple[str, list[str], dict[str, Any]]]] = {
    "momentum": {
        "rsi": ("rsi", ["rsi"], {"length": 14}),
        "macd": (
            "macd",
            ["macd", "signal", "hist"],
            {"fast_length": 12, "slow_length": 26, "signal_length": 9},
        ),
        "stochastic": (
            "stochastic",
            ["k", "d"],
            {"period_k": 5, "smooth_k": 3, "period_d": 3},
        ),
        "roc": ("roc", ["roc"], {"length": 1}),
        "momentum": ("momentum", ["momentum"], {"length": 1}),
        "williams_r": ("williams_r", ["williams_r"], {"length": 14}),
        "tsi": ("tsi", ["tsi"], {"long_length": 25, "short_length": 13}),
        "rmi": ("rmi", ["rmi"], {"length": 14, "momentum_length": 5}),
        "ppo": ("ppo", ["ppo"], {"fast_length": 12, "slow_length": 26}),
        "coppock_curve": (
            "coppock_curve",
            ["coppock"],
            {"wl1": 14, "wl2": 11, "wma_length": 10},
        ),
        "kst": ("kst", ["kst"], {}),
        "uo": ("uo", ["uo"], {"fast": 7, "medium": 14, "slow": 28}),
    },
    "trend": {
        "adx": ("adx", ["adx", "di_plus", "di_minus"], {"length": 14}),
        "supertrend": (
            "supertrend",
            ["supertrend", "direction"],
            {"length": 7, "multiplier": 3.0},
        ),
        "ichimoku": (
            "ichimoku_cloud",
            ["tenkan", "kijun", "senkou_a", "senkou_b"],
            {},
        ),
        "parabolic_sar": (
            "parabolic_sar",
            ["sar"],
            {"step": 0.02, "max_step": 0.2},
        ),
        "vortex": ("vortex_indicator", ["vi_plus", "vi_minus"], {"length": 14}),
        "trix": ("trix", ["trix"], {"length": 18}),
        "pfe": ("pfe", ["pfe"], {"n": 10, "m": 10}),
        "td_sequential": ("td_sequential", ["td"], {"length": 9}),
    },
    "volatility": {
        "atr": ("atr", ["atr"], {"length": 14}),
        "bollinger": (
            "bollinger_bands",
            ["upper", "middle", "lower"],
            {"length": 20, "num_std_dev": 2.0},
        ),
        "keltner": (
            "keltner_channels",
            ["upper", "middle", "lower"],
            {"length": 20, "multiplier": 2.0},
        ),
        "donchian": (
            "donchian_channels",
            ["upper", "middle", "lower"],
            {"length": 20},
        ),
        "chandelier": (
            "chandelier_exit",
            ["long_exit", "short_exit"],
            {"length": 22, "multiplier": 3.0},
        ),
        "choppiness": ("choppiness_index", ["choppiness"], {"length": 14}),
        "cci": ("cci", ["cci"], {"length": 14, "c": 0.015}),
        "vhf": ("vhf", ["vhf"], {"length": 28}),
        "ebsw": ("ebsw", ["ebsw"], {"length": 40, "bars": 10}),
    },
    "volume": {
        "obv": ("obv", ["obv"], {"length": 14, "initial": 1}),
        "obv_slope": ("obv_slope", ["obv_slope"], {"length": 20, "lookback": 10}),
        "mfi": ("mfi", ["mfi"], {"length": 14, "drift": 1}),
        "vwap": ("rolling_vwap", ["vwap"], {"length": 14}),
        "twap": ("twap", ["twap"], {"length": 14}),
        "cmf": ("chaikin_money_flow", ["cmf"], {"length": 20}),
        "force_index": ("force_index", ["force_index"], {"length": 13}),
        "pvt": ("pvt", ["pvt"], {"length": 14, "drift": 1}),
        "ad_line": ("accumulation_distribution_line", ["ad_line"], {}),
        "avg_quote_volume": (
            "average_quote_volume",
            ["avg_quote_volume"],
            {"window_size": 14},
        ),
    },
    "structure": {
        "support_resistance": (
            "support_resistance",
            ["support", "resistance"],
            {"length": 30},
        ),
        "fibonacci_retracement": (
            "fibonacci_retracement",
            ["fib"],
            {"length": 20},
        ),
        "fibonacci_bollinger_bands": (
            "fibonacci_bollinger_bands",
            ["upper", "middle", "lower"],
            {"length": 20, "mult": 3.0},
        ),
        "pivot_points": (
            "pivot_points",
            ["p", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"],
            {},
        ),
        "fibonacci_pivot_points": (
            "fibonacci_pivot_points",
            ["p", "r1", "s1", "r2", "s2", "r3", "s3"],
            {},
        ),
    },
    "statistical": {
        "hurst": ("hurst", ["hurst"], {"max_lag": 20}),
        "zscore": ("zscore", ["zscore"], {"length": 30, "std": 1.0}),
        "entropy": ("entropy", ["entropy"], {"length": 10, "base": 2.0}),
        "kurtosis": ("kurtosis", ["kurtosis"], {"length": 30}),
        "skew": ("skew", ["skew"], {"length": 30}),
        "stdev": ("stdev", ["stdev"], {"length": 30, "ddof": 1}),
        "variance": ("variance", ["variance"], {"length": 30, "ddof": 1}),
        "quantile": ("quantile", ["quantile"], {"length": 30, "q": 0.5}),
        "mad": ("mad", ["mad"], {"length": 30}),
        "linreg": ("linreg", ["linreg"], {"length": 14, "r": False}),
        "fear_and_greed": ("fear_and_greed_index", ["fng"], {}),
    },
}


def available(category: str) -> list[str]:
    """List the indicator names MCP exposes for ``category``."""
    return sorted(_CATALOG.get(category, {}).keys())


def run_indicator(
    ti: TechnicalIndicators,
    category: str,
    indicator: str,
    params: dict[str, Any] | None,
) -> tuple[list[tuple[str, list[float | None]]], dict[str, Any]]:
    """Run ``category.indicator`` on a loaded ``TechnicalIndicators`` handle.

    Returns ``(series, merged_params)`` where ``series`` is a list of
    ``(label, values)`` tuples — one entry per output component (e.g. MACD
    emits three). NaNs in the underlying ndarrays become ``None`` so the
    result is JSON-safe.
    """
    try:
        method_name, labels, defaults = _CATALOG[category][indicator]
    except KeyError as exc:
        raise ValueError(
            f"unknown indicator '{indicator}' for category '{category}'"
        ) from exc

    merged = dict(defaults)
    for key, value in (params or {}).items():
        default = defaults.get(key)
        if isinstance(default, bool):
            merged[key] = bool(value)
        elif isinstance(default, int):
            merged[key] = int(value)
        else:
            merged[key] = value
    method = getattr(ti, method_name)
    raw = method(**merged)

    components: list[np.ndarray]
    if isinstance(raw, tuple):
        components = list(raw)
    else:
        components = [raw]

    # Some facade methods return fewer components than labels (e.g. some
    # ichimoku builds). Zip against the shorter; callers see exactly what's
    # produced.
    series: list[tuple[str, list[float | None]]] = []
    for label, arr in zip(labels, components):
        series.append((label, _to_json_list(arr)))
    return series, merged


def _to_json_list(arr: Any) -> list[float | None]:
    """Convert an ndarray / iterable into a NaN-safe list of floats."""
    if arr is None:
        return []
    flat = np.asarray(arr, dtype=float).ravel()
    out: list[float | None] = []
    for value in flat:
        if math.isnan(value) or math.isinf(value):
            out.append(None)
        else:
            out.append(float(value))
    return out
