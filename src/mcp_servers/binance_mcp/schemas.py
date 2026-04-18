"""Pydantic response models for the Binance MCP server.

Inputs are validated at the handler signature via ``Annotated[...,
Field(...)]``. Response models here keep every tool's shape explicit so
MCP clients can discover the schema.
"""

from typing import Literal

from pydantic import BaseModel


Timeframe = Literal[
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
]


class OHLCVBar(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(BaseModel):
    success: bool = True
    symbol: str
    timeframe: str
    count: int
    candles: list[OHLCVBar]
    freshness_seconds: int | None = None


class TickerResponse(BaseModel):
    success: bool = True
    symbol: str
    last_price: float
    price_change_percent: float
    bid_price: float
    ask_price: float
    quote_volume: float
    close_time: int
    freshness_seconds: int | None = None


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookResponse(BaseModel):
    success: bool = True
    symbol: str
    best_bid: OrderBookLevel | None = None
    best_ask: OrderBookLevel | None = None
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    last_update_id: int


class FundingRateResponse(BaseModel):
    success: bool = True
    symbol: str
    mark_price: float | None = None
    index_price: float | None = None
    last_funding_rate: float | None = None
    next_funding_time: int | None = None
    freshness_seconds: int | None = None


class OpenInterestResponse(BaseModel):
    success: bool = True
    symbol: str
    open_interest: float
    timestamp: int
    freshness_seconds: int | None = None


class BalanceAsset(BaseModel):
    asset: str
    free: float
    locked: float


class BalanceResponse(BaseModel):
    success: bool = True
    balances: list[BalanceAsset]


class PositionItem(BaseModel):
    symbol: str
    position_amount: float
    entry_price: float
    mark_price: float | None = None
    unrealized_pnl: float | None = None
    leverage: float | None = None
    side: str


class PositionsResponse(BaseModel):
    success: bool = True
    symbol: str
    positions: list[PositionItem]


class OrderStatusResponse(BaseModel):
    success: bool = True
    symbol: str
    order_id: int
    status: str
    side: str
    type: str
    price: float
    orig_qty: float
    executed_qty: float
    reduce_only: bool
    update_time: int


class CancelOrderResponse(BaseModel):
    success: bool = True
    symbol: str
    order_id: int
    status: str
    side: str
    type: str
    orig_qty: float
