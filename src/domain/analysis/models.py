"""Analysis domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Signal(str, Enum):
    """Directional signal used by the technical analyzer."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass(slots=True)
class IndicatorSet:
    """Calculated technical indicators for a symbol."""

    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    ema_20: float
    ema_50: float
    volume_sma_20: float
    # Extended indicators (have defaults so existing code continues to work)
    atr: float = 0.0            # Average True Range (absolute price units)
    adx: float = 0.0            # Average Directional Index (0-100)
    obv_slope: float = 0.0      # On-Balance Volume linear slope (normalised)
    choppiness: float = 50.0    # Choppiness Index (0-100; >61.8 = choppy market)


@dataclass(slots=True)
class TechnicalAnalysis:
    """Technical analysis result for a symbol."""

    symbol: str
    signal: Signal
    indicators: IndicatorSet
    reasoning: str


@dataclass(slots=True)
class PatternResult:
    """Detected chart patterns and optional chart image."""

    symbol: str
    patterns: list[str] = field(default_factory=list)
    support: float | None = None
    resistance: float | None = None
    chart_png_b64: str = ""
