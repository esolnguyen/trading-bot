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
