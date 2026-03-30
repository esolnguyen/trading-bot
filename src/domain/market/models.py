"""Market domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OHLCVCandle:
    """Single OHLCV candle in unix milliseconds."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class MarketSnapshot:
    """Normalized market snapshot used by the trading loop."""

    symbol: str
    price: float
    change_24h_pct: float
    volume_24h: float
    bid: float
    ask: float
    candles: list[OHLCVCandle] = field(default_factory=list)
    timestamp: int = 0
    # Futures-only enrichment (None for spot or when unavailable)
    funding_rate: float | None = None       # current 8h funding rate, e.g. 0.0001 = 0.01%
    open_interest: float | None = None      # notional open interest in USD
