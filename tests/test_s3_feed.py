"""Acceptance tests for S3 market data ingestion."""

from __future__ import annotations

import asyncio

from src.core.config import Settings
from src.domain.market import MarketSnapshot
from src.infrastructure.binance import BinanceFeed
from src.services.analysis import MarketAggregator


def build_settings() -> Settings:
    return Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
    )


class FakeApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def get_ticker(self, symbol: str) -> dict:
        self.calls.append(("get_ticker", {"symbol": symbol}))
        return {
            "symbol": symbol,
            "priceChangePercent": "1.25",
            "lastPrice": "64000.0",
            "bidPrice": "63999.0",
            "askPrice": "64001.0",
            "quoteVolume": "125000000.0",
            "closeTime": 1700000000000,
        }

    def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        self.calls.append(("get_order_book", {"symbol": symbol, "limit": limit}))
        return {
            "lastUpdateId": 1700000000123,
            "bids": [["63998.5", "1.2"]],
            "asks": [["64001.5", "0.8"]],
        }

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 200) -> list[list[float]]:
        self.calls.append(("get_klines", {"symbol": symbol, "interval": interval, "limit": limit}))
        assert symbol.endswith("USDT")
        assert interval == "1h"
        assert limit == 200
        return [[1700000000000 + i, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(limit)]

    def get_account(self) -> dict:
        self.calls.append(("get_account", {}))
        return {"balances": [{"asset": "USDT", "free": "1000", "locked": "0"}]}

    def close_connection(self) -> None:
        self.closed = True


class RateLimitExceeded(Exception):
    """Local stand-in for ccxt.RateLimitExceeded in tests."""


class RetryingClient(FakeApiClient):
    def __init__(self) -> None:
        super().__init__()
        self._first = True

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 200) -> list[list[float]]:
        if self._first:
            self._first = False
            raise RateLimitExceeded("slow down")
        return super().get_klines(symbol, interval=interval, limit=limit)


class ErrorApiClient(FakeApiClient):
    def get_ticker(self, symbol: str) -> dict:
        raise RuntimeError("upstream failure")


class FuturesStyleTickerClient(FakeApiClient):
    def get_ticker(self, symbol: str) -> dict:
        self.calls.append(("get_ticker", {"symbol": symbol}))
        return {
            "symbol": symbol,
            "priceChangePercent": "1.25",
            "lastPrice": "64000.0",
            "quoteVolume": "125000000.0",
            "closeTime": 1700000000000,
        }


def test_snapshot_returns_market_snapshot_with_200_candles() -> None:
    feed = BinanceFeed(
        build_settings(),
        api_client_factory=lambda: FakeApiClient(),
    )
    aggregator = MarketAggregator(feed)

    snapshot = asyncio.run(aggregator.snapshot("BTCUSDT"))
    asyncio.run(feed.close())

    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.symbol == "BTCUSDT"
    assert len(snapshot.candles) == 200
    assert snapshot.bid == 63998.5
    assert snapshot.ask == 64001.5


def test_snapshot_returns_valid_eth_snapshot_independently() -> None:
    feed = BinanceFeed(
        build_settings(),
        api_client_factory=lambda: FakeApiClient(),
    )
    aggregator = MarketAggregator(feed)

    snapshot = asyncio.run(aggregator.snapshot("ETHUSDT"))
    asyncio.run(feed.close())

    assert snapshot.symbol == "ETHUSDT"
    assert len(snapshot.candles) == 200


def test_rate_limit_is_retried_once(monkeypatch) -> None:
    client = RetryingClient()
    feed = BinanceFeed(
        build_settings(),
        api_client_factory=lambda: client,
    )

    sleep_calls: list[int | float] = []

    async def fake_sleep(delay: int | float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("src.infrastructure.binance.feed.asyncio.sleep", fake_sleep)

    candles = asyncio.run(feed.get_ohlcv("BTCUSDT"))
    asyncio.run(feed.close())

    assert len(candles) == 200
    assert sleep_calls == [10]
    assert any(call[0] == "get_klines" for call in client.calls)


def test_api_error_raises_runtime_error() -> None:
    feed = BinanceFeed(
        build_settings(),
        api_client_factory=lambda: ErrorApiClient(),
    )

    try:
        asyncio.run(feed.get_ticker("BTCUSDT"))
    except RuntimeError as exc:
        assert "upstream failure" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for API error response")
    finally:
        asyncio.run(feed.close())


def test_futures_style_ticker_without_bid_ask_still_maps_snapshot() -> None:
    feed = BinanceFeed(
        build_settings(),
        api_client_factory=lambda: FuturesStyleTickerClient(),
    )
    aggregator = MarketAggregator(feed)

    snapshot = asyncio.run(aggregator.snapshot("BTCUSDT"))
    asyncio.run(feed.close())

    assert snapshot.price == 64000.0
    assert snapshot.bid == 63998.5
    assert snapshot.ask == 64001.5
