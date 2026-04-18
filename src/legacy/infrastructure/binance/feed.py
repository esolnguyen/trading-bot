"""Binance UM futures market data feed backed by the SDK."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from binance.error import ClientError

from src.mcp_servers.config import Settings
from src.mcp_servers.shared.domain.market import OHLCVCandle
from src.mcp_servers.shared.infrastructure.binance.client import (
    build_client,
    close_client,
)


class BinanceFeed:
    """Fetch market & account data from Binance UM futures through the SDK."""

    def __init__(
        self,
        settings: Settings,
        *,
        api_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._api_client_factory = api_client_factory or (
            lambda: build_client(settings)
        )
        self._api_client: Any | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "BinanceFeed":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        await self._ensure_api_client()

    async def close(self) -> None:
        async with self._lock:
            if self._api_client is not None:
                close_client(self._api_client)
                self._api_client = None

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 200
    ) -> list[OHLCVCandle]:
        client = await self._ensure_api_client()
        for attempt in range(2):
            try:
                rows = await asyncio.to_thread(
                    client.klines, symbol, timeframe, limit=limit
                )
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
            except ClientError as exc:
                if _is_rate_limit(exc) and attempt == 0:
                    await asyncio.sleep(10)
                    continue
                raise RuntimeError(
                    f"Failed to fetch OHLCV for {symbol}: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to fetch OHLCV for {symbol}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Failed to fetch OHLCV for {symbol}: rate limit retry exhausted"
        )

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        client = await self._ensure_api_client()
        try:
            ticker = await asyncio.to_thread(
                client.ticker_24hr_price_change, symbol=symbol
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch ticker for {symbol}: {exc}") from exc
        last_price = float(ticker["lastPrice"])
        close_time = int(ticker.get("closeTime") or ticker.get("time") or 0)
        # Futures 24h ticker has no bid/ask fields; default to last price.
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
        client = await self._ensure_api_client()
        for attempt in range(3):
            try:
                order_book = await asyncio.to_thread(client.depth, symbol, limit=limit)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Failed to fetch order book for {symbol}: {exc}"
                ) from exc
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
        """Return current funding rate and next funding time.

        Returns ``{"success": False}`` on any error so callers can treat it as
        an optional enrichment.
        """
        client = await self._ensure_api_client()
        try:
            data = await asyncio.to_thread(client.mark_price, symbol=symbol)
            return {"success": True, "data": data}
        except Exception:  # noqa: BLE001
            return {"success": False}

    async def get_open_interest(self, symbol: str) -> dict[str, Any]:
        client = await self._ensure_api_client()
        try:
            data = await asyncio.to_thread(client.open_interest, symbol)
            return {"success": True, "data": data}
        except Exception:  # noqa: BLE001
            return {"success": False}

    async def get_order_status(self, symbol: str, order_id: int) -> dict[str, Any]:
        client = await self._ensure_api_client()
        try:
            data = await asyncio.to_thread(
                client.query_order, symbol=symbol, orderId=order_id
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch order {order_id}: {exc}") from exc
        return {"success": True, "data": data, "timestamp": 0}

    async def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        client = await self._ensure_api_client()
        try:
            data = await asyncio.to_thread(
                client.cancel_order, symbol=symbol, orderId=order_id
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to cancel order {order_id}: {exc}") from exc
        return {"success": True, "data": data, "timestamp": 0}

    async def get_open_positions(self, symbol: str) -> list[dict[str, Any]]:
        client = await self._ensure_api_client()
        try:
            data = await asyncio.to_thread(client.get_position_risk, symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to fetch positions for {symbol}: {exc}"
            ) from exc
        return data if isinstance(data, list) else []

    async def get_balance(self) -> dict[str, Any]:
        client = await self._ensure_api_client()
        try:
            account = await asyncio.to_thread(client.account)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch account balance: {exc}") from exc
        return {"success": True, "data": _normalize_account(account), "timestamp": 0}

    async def _ensure_api_client(self) -> Any:
        async with self._lock:
            if self._api_client is not None:
                return self._api_client
            self._api_client = self._api_client_factory()
            return self._api_client


def _is_rate_limit(exc: ClientError) -> bool:
    return getattr(exc, "status_code", None) in (418, 429)


def _normalize_account(account: dict[str, Any]) -> dict[str, Any]:
    """Map the UM futures `/fapi/v2/account` shape into {'balances': [...]}."""
    if "balances" in account:
        return account
    if "assets" in account:
        return {
            "balances": [
                {
                    "asset": asset.get("asset"),
                    "free": asset.get(
                        "availableBalance", asset.get("walletBalance", "0")
                    ),
                    "locked": asset.get("initialMargin", "0"),
                }
                for asset in account.get("assets", [])
                if asset.get("asset")
            ]
        }
    return account
