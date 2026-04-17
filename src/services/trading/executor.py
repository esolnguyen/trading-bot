"""Trade execution through the local Binance REST client.

Heavy lifting lives in sibling modules:
  * :mod:`executor_filters` — LOT_SIZE / PRICE_FILTER caching + rounding
  * :mod:`executor_brackets` — SL/TP bracket placement + cancellation
  * :mod:`executor_limit`    — LIMIT fill polling + price extraction
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision, TradeOutcome
from src.infrastructure.binance.rest_client import BinanceRestClient
from src.services.trading.executor_brackets import (
    cancel_bracket_orders as _cancel_bracket_orders,
    place_bracket_orders as _place_bracket_orders,
)
from src.services.trading.executor_filters import ExchangeFilters
from src.services.trading.executor_limit import await_limit_fill, extract_price

logger = logging.getLogger(__name__)

# Map close actions to the Binance side that closes the position.
_CLOSE_SIDE = {
    Action.CLOSE_LONG: "SELL",
    Action.CLOSE_SHORT: "BUY",
}


class Executor:
    """Submit validated trade decisions through the local Binance REST client."""

    def __init__(
        self,
        settings: Settings,
        *,
        api_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._api_client_factory = (
            api_client_factory or self._default_api_client_factory
        )
        self._api_client: Any | None = None
        self._lock = asyncio.Lock()
        self._filters = ExchangeFilters()

    async def initialize(self) -> None:
        """Apply leverage and pre-warm exchange filters.

        Called automatically by TradingLoop.run() at startup.  Also called
        by ``__aenter__`` for callers that use the async-context-manager API.
        """
        client = await self._ensure_api_client()
        await self._apply_leverage(client)
        await self._filters.prewarm(
            getattr(self.settings, "trading_symbols", []), client
        )

    async def __aenter__(self) -> "Executor":
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._lock:
            if self._api_client is not None:
                close_connection = getattr(self._api_client, "close_connection", None)
                if callable(close_connection):
                    close_connection()
                self._api_client = None

    @staticmethod
    def _simulated_outcome(decision: TradeDecision, *, dry_run: bool) -> TradeOutcome:
        """Return a no-side-effect TradeOutcome (for HOLD, dry runs, or rounded-to-zero qty)."""
        return TradeOutcome(
            decision=decision,
            order_id=None,
            executed_price=None,
            pnl_usdt=None,
            dry_run=dry_run,
            timestamp=decision.timestamp,
        )

    async def execute(self, decision: TradeDecision, dry_run: bool) -> TradeOutcome:
        if dry_run or decision.action == Action.HOLD:
            return self._simulated_outcome(decision, dry_run=dry_run)

        client = await self._ensure_api_client()
        qty_str = await self._filters.format_quantity(
            decision.symbol, decision.quantity, client,
            order_type=decision.order_type,
        )
        if float(qty_str) <= 0:
            logger.warning(
                "Skipping order for %s: quantity %.8f rounds to zero (step_size too large)",
                decision.symbol,
                decision.quantity,
            )
            return self._simulated_outcome(decision, dry_run=True)

        side = _CLOSE_SIDE.get(decision.action, decision.action.value)
        is_close = decision.action in _CLOSE_SIDE
        params: dict[str, Any] = {
            "symbol": decision.symbol,
            "side": side,
            "type": decision.order_type,
            "quantity": qty_str,
        }
        if is_close and getattr(self.settings, "binance_product", "spot") == "usdt_futures":
            params["reduceOnly"] = "true"
        if decision.price is not None:
            params["price"] = await self._filters.format_price(
                decision.symbol, decision.price, client
            )
        logger.info(
            "Submitting order: %s (stepSize=%s, marketStepSize=%s, tickSize=%s)",
            params,
            self._filters.step_size_cache.get(decision.symbol, "?"),
            self._filters.market_step_size_cache.get(decision.symbol, "?"),
            self._filters.tick_size_cache.get(decision.symbol, "?"),
        )
        try:
            data = client.create_order(**params)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(str(exc)) from exc
        order_id = str(data.get("orderId", ""))
        executed_price = extract_price(data)
        logger.info(
            "Order placed: %s %s qty=%.8f order_id=%s",
            decision.action.value,
            decision.symbol,
            decision.quantity,
            order_id,
        )
        outcome = TradeOutcome(
            decision=decision,
            order_id=order_id,
            executed_price=executed_price,
            pnl_usdt=None,
            dry_run=False,
            timestamp=int(data.get("transactTime", decision.timestamp)),
        )

        if decision.order_type == "LIMIT" and order_id:
            timeout = getattr(self.settings, "limit_order_timeout_seconds", 300)
            outcome = await await_limit_fill(outcome, client, timeout_seconds=timeout)

        return outcome

    async def place_bracket_orders(
        self,
        symbol: str,
        entry_side: str,
        sl_price: float,
        tp_price: float,
        quantity: float = 0.0,
    ) -> tuple[str | None, str | None]:
        client = await self._ensure_api_client()
        return await _place_bracket_orders(
            client, self.settings, self._filters,
            symbol, entry_side, sl_price, tp_price, quantity,
        )

    async def cancel_bracket_orders(
        self,
        symbol: str,
        sl_order_id: str | None,
        tp_order_id: str | None,
    ) -> None:
        if not sl_order_id and not tp_order_id:
            return
        client = await self._ensure_api_client()
        await _cancel_bracket_orders(client, symbol, sl_order_id, tp_order_id)

    async def get_live_position_size(self, symbol: str) -> float:
        """Return the absolute open position quantity on the exchange.

        Returns ``float('inf')`` on error so that a transient API failure never
        incorrectly triggers a local position clear.
        """
        client = await self._ensure_api_client()
        try:
            positions = await asyncio.to_thread(client.get_open_positions, symbol)
            return sum(abs(float(p.get("positionAmt", 0))) for p in positions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_live_position_size failed for %s: %s", symbol, exc)
            return float("inf")

    async def _ensure_api_client(self) -> Any:
        async with self._lock:
            if self._api_client is not None:
                return self._api_client
            self._api_client = self._api_client_factory()
            return self._api_client

    async def _apply_leverage(self, client: Any) -> None:
        """Set configured leverage on all trading symbols (futures only)."""
        if getattr(self.settings, "binance_product", "spot") != "usdt_futures":
            return
        leverage = getattr(self.settings, "futures_leverage", 1)
        if leverage == 1:
            return
        symbols = getattr(self.settings, "trading_symbols", [])
        for symbol in symbols:
            try:
                await asyncio.to_thread(client.change_leverage, symbol, leverage)
                logger.info("Leverage set to %dx for %s", leverage, symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set leverage for %s: %s", symbol, exc)

    def _default_api_client_factory(self) -> Any:
        return BinanceRestClient(
            api_key=self.settings.binance_api_key,
            api_secret=self.settings.binance_api_secret,
            product=self.settings.binance_product,
            testnet=self.settings.binance_testnet,
            base_url=self.settings.binance_base_url,
        )
