"""Tool handlers for the Binance MCP server.

Handlers are thin: validate inputs via function signature, call
``BinanceToolsService``, map the raw SDK shape onto the matching
pydantic response model.

Note: do NOT add ``from __future__ import annotations`` — FastMCP
builds tool schemas via pydantic's TypeAdapter which cannot resolve
stringified hints.
"""

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from src.mcp_servers.base import tool_error
from src.mcp_servers.binance_mcp.schemas import (
    BalanceAsset,
    BalanceResponse,
    CancelOrderResponse,
    FundingRateResponse,
    OHLCVBar,
    OHLCVResponse,
    OpenInterestResponse,
    OrderBookLevel,
    OrderBookResponse,
    OrderStatusResponse,
    PositionItem,
    PositionsResponse,
    TickerResponse,
    Timeframe,
)
from src.mcp_servers.binance_mcp.service import BinanceToolsService

logger = logging.getLogger(__name__)


_SymbolField = Annotated[
    str,
    Field(
        min_length=3,
        max_length=20,
        description="Binance UM-futures perpetual symbol, e.g. BTCUSDT.",
    ),
]


def register(mcp: FastMCP, service: BinanceToolsService) -> None:
    """Register every Binance tool against ``mcp``, bound to ``service``."""

    # ── market data ──────────────────────────────────────────────────────

    @mcp.tool()
    async def get_ohlcv(
        symbol: _SymbolField = "BTCUSDT",
        timeframe: Timeframe = "1h",
        limit: Annotated[int, Field(ge=1, le=1500)] = 200,
    ) -> dict:
        """Closed OHLCV candles for ``symbol`` on ``timeframe`` (most recent last)."""
        try:
            feed = await service.get_feed()
            candles = await feed.get_ohlcv(symbol, timeframe, limit=limit)
            if not candles:
                return tool_error("no_candles", symbol=symbol, timeframe=timeframe)
            return OHLCVResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                count=len(candles),
                candles=[
                    OHLCVBar(
                        timestamp=c.timestamp,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                    )
                    for c in candles
                ],
                freshness_seconds=service.freshness_seconds_ms(candles[-1].timestamp),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_ohlcv failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_ticker(symbol: _SymbolField = "BTCUSDT") -> dict:
        """24h ticker summary (last, change%, bid/ask, quote volume)."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_ticker(symbol)
            if not raw.get("success"):
                return tool_error("ticker_unavailable", symbol=symbol)
            data = raw.get("data") or {}
            return TickerResponse(
                symbol=str(data.get("symbol", symbol)).upper(),
                last_price=float(data.get("last_price", 0.0)),
                price_change_percent=float(data.get("price_change_percent", 0.0)),
                bid_price=float(data.get("bid_price", 0.0)),
                ask_price=float(data.get("ask_price", 0.0)),
                quote_volume=float(data.get("quote_volume", 0.0)),
                close_time=int(data.get("close_time") or 0),
                freshness_seconds=service.freshness_seconds_ms(data.get("close_time")),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_ticker failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_order_book(
        symbol: _SymbolField = "BTCUSDT",
        depth: Annotated[int, Field(ge=5, le=1000)] = 20,
    ) -> dict:
        """Current order book snapshot up to ``depth`` levels per side."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_order_book(symbol, limit=depth)
            if not raw.get("success"):
                return tool_error("order_book_unavailable", symbol=symbol)
            data = raw.get("data") or {}
            best_bid = data.get("bestBid")
            best_ask = data.get("bestAsk")
            return OrderBookResponse(
                symbol=symbol.upper(),
                best_bid=OrderBookLevel(**best_bid) if best_bid else None,
                best_ask=OrderBookLevel(**best_ask) if best_ask else None,
                bids=[OrderBookLevel(**b) for b in (data.get("bids") or [])],
                asks=[OrderBookLevel(**a) for a in (data.get("asks") or [])],
                last_update_id=int(raw.get("timestamp", 0)),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_order_book failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_funding_rate(symbol: _SymbolField = "BTCUSDT") -> dict:
        """Current mark/index price and most-recent funding rate for ``symbol``."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_funding_rate(symbol)
            if not raw.get("success"):
                return tool_error("funding_unavailable", symbol=symbol)
            data = raw.get("data") or {}
            event_time = int(data.get("time") or data.get("nextFundingTime") or 0)
            return FundingRateResponse(
                symbol=str(data.get("symbol", symbol)).upper(),
                mark_price=_as_float(data.get("markPrice")),
                index_price=_as_float(data.get("indexPrice")),
                last_funding_rate=_as_float(data.get("lastFundingRate")),
                next_funding_time=_as_int(data.get("nextFundingTime")),
                freshness_seconds=service.freshness_seconds_ms(event_time),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_funding_rate failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_open_interest(symbol: _SymbolField = "BTCUSDT") -> dict:
        """Total open interest (base units) for ``symbol``."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_open_interest(symbol)
            if not raw.get("success"):
                return tool_error("open_interest_unavailable", symbol=symbol)
            data = raw.get("data") or {}
            ts = int(data.get("time") or 0)
            return OpenInterestResponse(
                symbol=str(data.get("symbol", symbol)).upper(),
                open_interest=float(data.get("openInterest", 0.0)),
                timestamp=ts,
                freshness_seconds=service.freshness_seconds_ms(ts),
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_open_interest failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    # ── account view ─────────────────────────────────────────────────────

    @mcp.tool()
    async def get_balance() -> dict:
        """Account balances (free + locked per asset) for the configured API key."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_balance()
            if not raw.get("success"):
                return tool_error("balance_unavailable")
            balances_raw = (raw.get("data") or {}).get("balances") or []
            balances = [
                BalanceAsset(
                    asset=str(row.get("asset", "")),
                    free=_as_float(row.get("free")) or 0.0,
                    locked=_as_float(row.get("locked")) or 0.0,
                )
                for row in balances_raw
                if row.get("asset")
            ]
            return BalanceResponse(balances=balances).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_balance failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_open_positions(symbol: _SymbolField = "BTCUSDT") -> dict:
        """Open positions on ``symbol`` (empty list when flat)."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_open_positions(symbol)
            positions: list[PositionItem] = []
            for row in raw or []:
                amount = _as_float(row.get("positionAmt")) or 0.0
                if amount == 0.0:
                    continue
                positions.append(
                    PositionItem(
                        symbol=str(row.get("symbol", symbol)).upper(),
                        position_amount=amount,
                        entry_price=_as_float(row.get("entryPrice")) or 0.0,
                        mark_price=_as_float(row.get("markPrice")),
                        unrealized_pnl=_as_float(row.get("unRealizedProfit")),
                        leverage=_as_float(row.get("leverage")),
                        side="long" if amount > 0 else "short",
                    )
                )
            return PositionsResponse(
                symbol=symbol.upper(), positions=positions
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_open_positions failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    @mcp.tool()
    async def get_order_status(
        symbol: _SymbolField = "BTCUSDT",
        order_id: Annotated[int, Field(ge=1, description="Exchange order id.")] = 0,
    ) -> dict:
        """Current state of a single order by ``order_id``."""
        try:
            feed = await service.get_feed()
            raw = await feed.get_order_status(symbol, order_id)
            if not raw.get("success"):
                return tool_error(
                    "order_status_unavailable", symbol=symbol, order_id=order_id
                )
            data = raw.get("data") or {}
            return _order_status_from_sdk(data, symbol).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_order_status failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    # ── safe write ───────────────────────────────────────────────────────

    @mcp.tool()
    async def cancel_order(
        symbol: _SymbolField = "BTCUSDT",
        order_id: Annotated[int, Field(ge=1, description="Exchange order id.")] = 0,
    ) -> dict:
        """Cancel an open order by ``order_id``.

        Blast-radius-safe: cancellation can only shrink exposure. Order
        placement stays gated until Phase 7.
        """
        try:
            feed = await service.get_feed()
            raw = await feed.cancel_order(symbol, order_id)
            if not raw.get("success"):
                return tool_error("cancel_failed", symbol=symbol, order_id=order_id)
            data = raw.get("data") or {}
            return CancelOrderResponse(
                symbol=str(data.get("symbol", symbol)).upper(),
                order_id=int(data.get("orderId", order_id)),
                status=str(data.get("status", "")),
                side=str(data.get("side", "")),
                type=str(data.get("type", "")),
                orig_qty=_as_float(data.get("origQty")) or 0.0,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("cancel_order failed")
            return tool_error(f"{type(exc).__name__}: {exc}")


def _order_status_from_sdk(data: dict[str, Any], symbol: str) -> OrderStatusResponse:
    return OrderStatusResponse(
        symbol=str(data.get("symbol", symbol)).upper(),
        order_id=int(data.get("orderId", 0)),
        status=str(data.get("status", "")),
        side=str(data.get("side", "")),
        type=str(data.get("type", "")),
        price=_as_float(data.get("price")) or 0.0,
        orig_qty=_as_float(data.get("origQty")) or 0.0,
        executed_qty=_as_float(data.get("executedQty")) or 0.0,
        reduce_only=bool(data.get("reduceOnly", False)),
        update_time=int(data.get("updateTime") or 0),
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
