"""
Indicators package for bot-mcp-analysis.

Provides technical analysis indicators organized by category.
All indicator computation functions are numba-optimized where available,
with a pure-Python fallback when numba is not installed.
"""

# Base classes
from src.mcp_servers.analysis_mcp.indicators.indicator_base import IndicatorBase
from src.mcp_servers.analysis_mcp.indicators.technical_indicators import (
    TechnicalIndicators,
)

# Constants
from src.mcp_servers.analysis_mcp.indicators.constants import INDICATOR_THRESHOLDS

# Momentum indicators
from src.mcp_servers.analysis_mcp.indicators.momentum import (
    rsi_numba,
    macd_numba,
    stochastic_numba,
    roc_numba,
    momentum_numba,
    williams_r_numba,
    tsi_numba,
    rmi_numba,
    ppo_numba,
    coppock_curve_numba,
    detect_rsi_divergence,
    calculate_relative_strength_numba,
    uo_numba,
    kst_numba,
)

# Overlap (moving average) indicators
from src.mcp_servers.analysis_mcp.indicators.overlap import (
    ema_numba,
    sma_numba,
    ewma_numba,
)

# Price transform indicators
from src.mcp_servers.analysis_mcp.indicators.price import (
    log_return_numba,
    percent_return_numba,
    pdist_numba,
)

# Sentiment indicators
from src.mcp_servers.analysis_mcp.indicators.sentiment import (
    FearGreedConfig,
    fear_and_greed_index_numba,
)

# Statistical indicators
from src.mcp_servers.analysis_mcp.indicators.statistical import (
    kurtosis_numba,
    skew_numba,
    stdev_numba,
    variance_numba,
    zscore_numba,
    mad_numba,
    quantile_numba,
    entropy_numba,
    hurst_numba,
    linreg_numba,
    apa_adaptive_eot_numba,
    calculate_eot_numba,
)

# Support/resistance indicators
from src.mcp_servers.analysis_mcp.indicators.support_resistance import (
    support_resistance_numba,
    find_support_resistance_numba,
    support_resistance_numba_advanced,
    advanced_support_resistance_numba,
    fibonacci_retracement_numba,
    floating_levels_numba,
    fibonacci_bollinger_bands_numba,
    pivot_points_numba,
    fibonacci_pivot_points_numba,
)

# Trend indicators
from src.mcp_servers.analysis_mcp.indicators.trend import (
    adx_numba,
    supertrend_numba,
    ichimoku_cloud_numba,
    parabolic_sar_numba,
    vortex_indicator_numba,
    trix_numba,
    pfe_numba,
    td_sequential_numba,
)

# Volatility indicators
from src.mcp_servers.analysis_mcp.indicators.volatility import (
    atr_numba,
    bollinger_bands_numba,
    chandelier_exit_numba,
    vhf_numba,
    ebsw_numba,
    keltner_channels_numba,
    donchian_channels_numba,
    choppiness_index_numba,
)

# Volume indicators
from src.mcp_servers.analysis_mcp.indicators.volume import (
    mfi_numba,
    obv_numba,
    obv_slope_numba,
    pvt_numba,
    chaikin_money_flow_numba,
    ad_line_numba,
    force_index_numba,
    eom_numba,
    volume_profile_numba,
    rolling_vwap_numba,
    twap_numba,
    average_quote_volume_numba,
    cci_numba,
)

__all__ = [
    # Base
    "IndicatorBase",
    "TechnicalIndicators",
    # Constants
    "INDICATOR_THRESHOLDS",
    # Momentum
    "rsi_numba",
    "macd_numba",
    "stochastic_numba",
    "roc_numba",
    "momentum_numba",
    "williams_r_numba",
    "tsi_numba",
    "rmi_numba",
    "ppo_numba",
    "coppock_curve_numba",
    "detect_rsi_divergence",
    "calculate_relative_strength_numba",
    "uo_numba",
    "kst_numba",
    # Overlap
    "ema_numba",
    "sma_numba",
    "ewma_numba",
    # Price
    "log_return_numba",
    "percent_return_numba",
    "pdist_numba",
    # Sentiment
    "fear_and_greed_index_numba",
    "FearGreedConfig",
    # Statistical
    "kurtosis_numba",
    "skew_numba",
    "stdev_numba",
    "variance_numba",
    "zscore_numba",
    "mad_numba",
    "quantile_numba",
    "entropy_numba",
    "hurst_numba",
    "linreg_numba",
    "apa_adaptive_eot_numba",
    "calculate_eot_numba",
    # Support/Resistance
    "support_resistance_numba",
    "find_support_resistance_numba",
    "support_resistance_numba_advanced",
    "advanced_support_resistance_numba",
    "fibonacci_retracement_numba",
    "floating_levels_numba",
    "fibonacci_bollinger_bands_numba",
    "pivot_points_numba",
    "fibonacci_pivot_points_numba",
    # Trend
    "adx_numba",
    "supertrend_numba",
    "ichimoku_cloud_numba",
    "parabolic_sar_numba",
    "vortex_indicator_numba",
    "trix_numba",
    "pfe_numba",
    "td_sequential_numba",
    # Volatility
    "atr_numba",
    "bollinger_bands_numba",
    "chandelier_exit_numba",
    "vhf_numba",
    "ebsw_numba",
    "keltner_channels_numba",
    "donchian_channels_numba",
    "choppiness_index_numba",
    # Volume
    "mfi_numba",
    "obv_numba",
    "obv_slope_numba",
    "pvt_numba",
    "chaikin_money_flow_numba",
    "ad_line_numba",
    "force_index_numba",
    "eom_numba",
    "volume_profile_numba",
    "rolling_vwap_numba",
    "twap_numba",
    "average_quote_volume_numba",
    "cci_numba",
]
