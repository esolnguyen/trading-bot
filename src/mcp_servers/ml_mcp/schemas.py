"""Pydantic response models for the ML MCP server.

Input validation is handled at the tool signature via ``Annotated[..., Field(...)]``
so FastMCP surfaces the constraints to clients. Response models are declared
here so every handler returns a validated, documented shape.
"""

from typing import Literal

from pydantic import BaseModel


Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
Direction = Literal["bullish", "bearish"]
Conviction = Literal["high", "moderate", "weak"]
Side = Literal["long", "short"]
SentimentLabel = Literal["bullish", "bearish", "neutral"]


class DirectionResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    probability: float
    direction: Direction
    conviction: Conviction
    reference_price: float
    freshness_seconds: int
    model_version: str


class AnomalyResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    is_anomaly: bool
    reference_price: float
    freshness_seconds: int
    model_version: str


class CycleResponse(BaseModel):
    success: bool = True
    symbol: str
    regime: str
    confidence: float
    daily_candles_used: int
    model_version: str


class KeyLevel(BaseModel):
    type: str
    center: float
    low: float
    high: float
    touches: int
    distance_pct: float


class KeyLevelsResponse(BaseModel):
    success: bool = True
    symbol: str
    reference_price: float
    levels: list[KeyLevel]
    freshness_seconds: int


class PercentileResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    reference_price: float
    summary: str
    freshness_seconds: int


class MarketConditions(BaseModel):
    rsi: float
    adx: float
    atr_percentage: float
    trend_direction: str
    volume_state: str
    bb_position: str


class OutcomeResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    side: Side
    win_probability: float
    reference_price: float
    market_conditions: MarketConditions
    blocked_by_threshold: bool
    model_version: str


class SentimentResponse(BaseModel):
    success: bool = True
    score: float
    label: SentimentLabel
    text_length: int
    model_version: str
    truncated: bool | None = None
