"""Binance market data feed backed by the local REST client."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from src.core.config import Settings
from src.domain.market import OHLCVCandle
from src.infrastructure.binance.rest_client import BinanceRestClient


class BinanceFeed:
    """Fetch market data from Binance through the local REST client."""

    def __init__(
        self,
        settings: Settings,
        *,
        api_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._api_client_factory = api_client_factory or self._default_api_client_factory
        self._api_client: Any | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "BinanceFeed":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Warm up the API client."""
        await self._ensure_api_client()

    async def close(self) -> None:
        """Close any open API client resources."""
        async with self._lock:
            if self._api_client is not None:
                close_connection = getattr(self._api_client, "close_connection", None)
                if callable(close_connection):
                    close_connection()
                self._api_client = None

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list[OHLCVCandle]:
        """Fetch OHLCV candles through the Binance client."""
        client = await self._ensure_api_client()

        for attempt in range(2):
            try:
                rows = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
                return [
                    OHLCVCandle(
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                    for row in rows
                ]
            except Exception as exc:  # noqa: BLE001
                if self._is_rate_limit_error(exc) and attempt == 0:
                    await asyncio.sleep(10)
                    continue
                raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {exc}") from exc

        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: rate limit retry exhausted")

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch 24h ticker data directly from Binance."""
        client = await self._ensure_api_client()
        try:
            ticker = client.get_ticker(symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch ticker for {symbol}: {exc}") from exc
        last_price = float(ticker["lastPrice"])
        close_time = int(ticker.get("closeTime") or ticker.get("time") or 0)
        return {
            "success": True,
            "data": {
                "symbol": ticker["symbol"],
                "price_change_percent": float(ticker["priceChangePercent"]),
                "last_price": last_price,
                "bid_price": float(ticker.get("bidPrice", last_price)),
                "ask_price": float(ticker.get("askPrice", last_price)),
                "quote_volume": float(ticker["quoteVolume"]),
                "close_time": close_time,
            },
            "timestamp": close_time,
        }

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch the current order book directly from Binance."""
        client = await self._ensure_api_client()
        for attempt in range(3):
            try:
                order_book = client.get_order_book(symbol=symbol, limit=limit)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to fetch order book for {symbol}: {exc}") from exc
        bids = [
            {"price": float(price), "quantity": float(quantity)}
            for price, quantity in order_book.get("bids", [])
        ]
        asks = [
            {"price": float(price), "quantity": float(quantity)}
            for price, quantity in order_book.get("asks", [])
        ]
        return {
            "success": True,
            "data": {
                "bestBid": bids[0] if bids else None,
                "bestAsk": asks[0] if asks else None,
                "bids": bids,
                "asks": asks,
            },
            "timestamp": int(order_book.get("lastUpdateId", 0)),
        }

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Futures only: fetch current funding rate and next funding time.

        Returns ``{"success": False}`` for spot products or on any error so
        callers can treat it as an optional enrichment.
        """
        if getattr(self.settings, "binance_product", "spot") != "usdt_futures":
            return {"success": False, "reason": "spot"}
        client = await self._ensure_api_client()
        try:
            data = client.get_funding_rate(symbol)
            return {"success": True, "data": data}
        except Exception:  # noqa: BLE001
            return {"success": False}

    async def get_open_interest(self, symbol: str) -> dict[str, Any]:
        """Futures only: fetch current open interest.

        Returns ``{"success": False}`` for spot products or on any error.
        """
        if getattr(self.settings, "binance_product", "spot") != "usdt_futures":
            return {"success": False, "reason": "spot"}
        client = await self._ensure_api_client()
        try:
            data = client.get_open_interest(symbol)
            return {"success": True, "data": data}
        except Exception:  # noqa: BLE001
            return {"success": False}

    async def get_order_status(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Fetch the current status of an order."""
        client = await self._ensure_api_client()
        try:
            data = client.get_order(symbol, order_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch order {order_id}: {exc}") from exc
        return {"success": True, "data": data, "timestamp": 0}

    async def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Cancel an open order."""
        client = await self._ensure_api_client()
        try:
            data = client.cancel_order(symbol, order_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to cancel order {order_id}: {exc}") from exc
        return {"success": True, "data": data, "timestamp": 0}

    async def get_open_positions(self, symbol: str) -> list[dict[str, Any]]:
        """Futures only: return position risk entries for the given symbol."""
        client = await self._ensure_api_client()
        try:
            data = client.get_open_positions(symbol)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch positions for {symbol}: {exc}") from exc
        return data if isinstance(data, list) else []

    async def get_balance(self) -> dict[str, Any]:
        """Fetch account balance directly from Binance."""
        client = await self._ensure_api_client()
        try:
            account = client.get_account()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch account balance: {exc}") from exc
        return {"success": True, "data": self._normalize_account(account), "timestamp": 0}

    async def _ensure_api_client(self) -> Any:
        async with self._lock:
            if self._api_client is not None:
                return self._api_client

            self._api_client = self._api_client_factory()
            return self._api_client

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        return exc.__class__.__name__ == "RateLimitExceeded"

    @staticmethod
    def _normalize_account(account: dict[str, Any]) -> dict[str, Any]:
        if "balances" in account:
            return account
        if "assets" in account:
            return {
                "balances": [
                    {
                        "asset": asset.get("asset"),
                        "free": asset.get("availableBalance", asset.get("walletBalance", "0")),
                        "locked": asset.get("initialMargin", "0"),
                    }
                    for asset in account.get("assets", [])
                    if asset.get("asset")
                ]
            }
        return account

    def _default_api_client_factory(self) -> Any:
        return BinanceRestClient(
            api_key=self.settings.binance_api_key,
            api_secret=self.settings.binance_api_secret,
            product=self.settings.binance_product,
            testnet=self.settings.binance_testnet,
            base_url=self.settings.binance_base_url,
        )
