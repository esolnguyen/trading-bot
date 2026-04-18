"""Aggregate raw market data into normalized snapshots."""

from __future__ import annotations

import asyncio

from src.mcp_servers.shared.domain.market import MarketSnapshot
from src.mcp_servers.shared.infrastructure.binance.feed import BinanceFeed


class MarketAggregator:
    """Map raw feed responses into shared market contracts."""

    def __init__(self, feed: BinanceFeed) -> None:
        self.feed = feed

    async def snapshot(
        self, symbol: str, timeframe: str = "1h", limit: int = 200
    ) -> MarketSnapshot:
        ticker, order_book, candles, fr_resp, oi_resp = await asyncio.gather(
            self.feed.get_ticker(symbol),
            self.feed.get_order_book(symbol),
            self.feed.get_ohlcv(symbol, timeframe=timeframe, limit=limit),
            self.feed.get_funding_rate(symbol),
            self.feed.get_open_interest(symbol),
        )

        ticker_data = ticker["data"]
        order_book_data = order_book["data"]
        best_bid = order_book_data.get("bestBid", {})
        best_ask = order_book_data.get("bestAsk", {})

        funding_rate: float | None = None
        if fr_resp.get("success"):
            try:
                funding_rate = float(fr_resp["data"]["lastFundingRate"])
            except (KeyError, TypeError, ValueError):
                pass

        open_interest: float | None = None
        if oi_resp.get("success"):
            try:
                open_interest = float(oi_resp["data"]["openInterest"])
            except (KeyError, TypeError, ValueError):
                pass

        return MarketSnapshot(
            symbol=symbol,
            price=float(ticker_data["last_price"]),
            change_24h_pct=float(ticker_data["price_change_percent"]),
            volume_24h=float(ticker_data["quote_volume"]),
            bid=float(best_bid.get("price", ticker_data["bid_price"])),
            ask=float(best_ask.get("price", ticker_data["ask_price"])),
            candles=candles,
            timestamp=int(ticker.get("timestamp", ticker_data["close_time"])),
            funding_rate=funding_rate,
            open_interest=open_interest,
        )
