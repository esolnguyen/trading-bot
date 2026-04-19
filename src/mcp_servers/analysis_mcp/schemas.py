"""Pydantic response models for the Analysis MCP server.

Input validation is handled at tool signatures via ``Annotated[..., Field(...)]``
so FastMCP surfaces the constraints to clients. Response models live here so
every handler returns a validated, documented shape.
"""

from typing import Literal

from pydantic import BaseModel


Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
Signal = Literal["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]


class IndicatorSnapshot(BaseModel):
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
    atr: float
    adx: float
    obv_slope: float
    choppiness: float
    vol_ratio: float
    cci_14: float


class IndicatorsResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    reference_price: float
    indicators: IndicatorSnapshot
    freshness_seconds: int


class MultiTFResponse(BaseModel):
    success: bool = True
    symbol: str
    trading_timeframe: str
    summary: str


class PatternResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    patterns: list[str]
    support: float | None
    resistance: float | None


class ChartResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    image_png_b64: str
    candle_count: int


class IndicatorSeries(BaseModel):
    name: str
    latest: float | None
    values: list[float | None]


class CategoryIndicatorResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    category: str
    indicator: str
    params: dict[str, float | int | bool]
    series: list[IndicatorSeries]
    freshness_seconds: int


class SignalResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    reference_price: float
    signal: Signal
    reasoning: str
    indicators: IndicatorSnapshot
    market_conditions: dict[str, float | int | bool | str | None]
    freshness_seconds: int


class SnapshotResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    price: float
    change_24h_pct: float
    volume_24h: float
    bid: float
    ask: float
    timestamp: int
    candle_count: int
    funding_rate: float | None
    open_interest: float | None
    freshness_seconds: int


Direction = Literal["long_short", "long_only"]


class BacktestTradeModel(BaseModel):
    entry_ts: int
    exit_ts: int
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl_pct: float


class BacktestMetricsModel(BaseModel):
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    turnover: float
    time_in_market_pct: float


class BacktestAssumptions(BaseModel):
    fee_bps: float
    slippage_bps: float
    direction: Direction
    warmup_bars: int
    bars_evaluated: int


class BacktestSignalResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    start_ts: int
    end_ts: int
    metrics: BacktestMetricsModel
    trades: list[BacktestTradeModel]
    equity_curve: list[float]
    assumptions: BacktestAssumptions
    caveats: list[str]
    freshness_seconds: int
