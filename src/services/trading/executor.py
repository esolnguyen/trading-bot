"""Trade execution through the local Binance REST client."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Callable

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision, TradeOutcome
from src.infrastructure.binance.rest_client import BinanceRestClient

logger = logging.getLogger(__name__)


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
        self._step_size_cache: dict[str, str] = (
            {}
        )  # symbol -> stepSize string e.g. "0.00001"

    async def __aenter__(self) -> "Executor":
        client = await self._ensure_api_client()
        await self._apply_leverage(client)
        await self._prewarm_step_sizes(client)
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

    async def execute(self, decision: TradeDecision, dry_run: bool) -> TradeOutcome:
        if dry_run or decision.action == Action.HOLD:
            return TradeOutcome(
                decision=decision,
                order_id=None,
                executed_price=None,
                pnl_usdt=None,
                dry_run=dry_run,
                timestamp=decision.timestamp,
            )

        client = await self._ensure_api_client()
        qty_str = await self._format_quantity(
            decision.symbol, decision.quantity, client
        )
        if float(qty_str) <= 0:
            logger.warning(
                "Skipping order for %s: quantity %.8f rounds to zero (step_size too large)",
                decision.symbol,
                decision.quantity,
            )
            return TradeOutcome(
                decision=decision,
                order_id=None,
                executed_price=None,
                pnl_usdt=None,
                dry_run=True,
                timestamp=decision.timestamp,
            )
        params = {
            "symbol": decision.symbol,
            "side": decision.action.value,
            "type": decision.order_type,
            "quantity": qty_str,
        }
        if decision.price is not None:
            params["price"] = decision.price
        try:
            data = client.create_order(**params)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(str(exc)) from exc
        order_id = str(data.get("orderId", ""))
        executed_price = self._extract_price(data)
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

        # For LIMIT orders, poll until filled or timeout, then cancel if unfilled.
        if decision.order_type == "LIMIT" and order_id:
            outcome = await self._await_limit_fill(outcome, client)

        return outcome

    async def place_bracket_orders(
        self,
        symbol: str,
        entry_side: str,
        sl_price: float,
        tp_price: float,
        quantity: float = 0.0,
    ) -> tuple[str | None, str | None]:
        """Place exchange-native SL and TP orders to protect an open position.

        For **futures**: places STOP_MARKET + TAKE_PROFIT_MARKET with
        ``closePosition=true`` (no quantity needed).

        For **spot**: places STOP_LOSS_LIMIT + TAKE_PROFIT_LIMIT with the given
        ``quantity``.  A small offset below/above the trigger price is applied so
        the limit order has a realistic chance of filling in a fast-moving market.

        Returns ``(sl_order_id, tp_order_id)``; either may be None if the
        individual placement fails — the error is logged and the position monitor
        will fall back to software SL management.
        """
        client = await self._ensure_api_client()
        close_side = "SELL" if entry_side == "BUY" else "BUY"
        is_futures = getattr(self.settings, "binance_product", "spot") == "usdt_futures"
        sl_order_id: str | None = None
        tp_order_id: str | None = None

        if is_futures:
            # --- Futures: reduce-only market orders, no quantity required ---
            try:
                sl_data = client.create_order(
                    symbol=symbol,
                    side=close_side,
                    type="STOP_MARKET",
                    stopPrice=f"{sl_price:.2f}",
                    closePosition="true",
                )
                sl_order_id = str(sl_data.get("orderId", "")) or None
                logger.info(
                    "Futures SL bracket placed: %s stopPrice=%.2f order=%s",
                    symbol,
                    sl_price,
                    sl_order_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to place futures SL bracket for %s: %s", symbol, exc
                )

            try:
                tp_data = client.create_order(
                    symbol=symbol,
                    side=close_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=f"{tp_price:.2f}",
                    closePosition="true",
                )
                tp_order_id = str(tp_data.get("orderId", "")) or None
                logger.info(
                    "Futures TP bracket placed: %s stopPrice=%.2f order=%s",
                    symbol,
                    tp_price,
                    tp_order_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to place futures TP bracket for %s: %s", symbol, exc
                )

        else:
            # --- Spot: limit orders at a small offset from the trigger price ---
            if quantity <= 0:
                logger.warning(
                    "Spot bracket orders require quantity > 0 for %s — skipping (software SL active)",
                    symbol,
                )
                return None, None

            sl_offset = getattr(self.settings, "spot_sl_limit_offset_pct", 0.01)
            tp_offset = getattr(self.settings, "spot_tp_limit_offset_pct", 0.002)
            qty_str = f"{quantity:.8f}"

            # SL: sell limit placed below trigger so it has room to fill on the way down
            if close_side == "SELL":
                sl_limit_price = sl_price * (1.0 - sl_offset)
                tp_limit_price = tp_price * (1.0 - tp_offset)
            else:
                # Closing a short: BUY limit placed above trigger
                sl_limit_price = sl_price * (1.0 + sl_offset)
                tp_limit_price = tp_price * (1.0 + tp_offset)

            try:
                sl_data = client.create_order(
                    symbol=symbol,
                    side=close_side,
                    type="STOP_LOSS_LIMIT",
                    quantity=qty_str,
                    stopPrice=f"{sl_price:.2f}",
                    price=f"{sl_limit_price:.2f}",
                    timeInForce="GTC",
                )
                sl_order_id = str(sl_data.get("orderId", "")) or None
                logger.info(
                    "Spot SL bracket placed: %s stopPrice=%.2f limitPrice=%.2f qty=%s order=%s",
                    symbol,
                    sl_price,
                    sl_limit_price,
                    qty_str,
                    sl_order_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to place spot SL bracket for %s: %s", symbol, exc)

            try:
                tp_data = client.create_order(
                    symbol=symbol,
                    side=close_side,
                    type="TAKE_PROFIT_LIMIT",
                    quantity=qty_str,
                    stopPrice=f"{tp_price:.2f}",
                    price=f"{tp_limit_price:.2f}",
                    timeInForce="GTC",
                )
                tp_order_id = str(tp_data.get("orderId", "")) or None
                logger.info(
                    "Spot TP bracket placed: %s stopPrice=%.2f limitPrice=%.2f qty=%s order=%s",
                    symbol,
                    tp_price,
                    tp_limit_price,
                    qty_str,
                    tp_order_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to place spot TP bracket for %s: %s", symbol, exc)

        return sl_order_id, tp_order_id

    async def cancel_bracket_orders(
        self,
        symbol: str,
        sl_order_id: str | None,
        tp_order_id: str | None,
    ) -> None:
        """Cancel surviving SL/TP bracket orders.

        Called before a bot-managed close or when reconciliation detects that the
        exchange already filled one side.  Silently ignores already-filled orders.
        """
        if not sl_order_id and not tp_order_id:
            return
        client = await self._ensure_api_client()
        for oid, label in ((sl_order_id, "SL"), (tp_order_id, "TP")):
            if not oid:
                continue
            try:
                client.cancel_order(symbol, int(oid))
                logger.info("Cancelled %s bracket order %s for %s", label, oid, symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not cancel %s bracket %s for %s: %s", label, oid, symbol, exc
                )

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

    async def _await_limit_fill(
        self, outcome: TradeOutcome, client: Any
    ) -> TradeOutcome:
        """Poll until a LIMIT order is filled or the timeout elapses, then cancel."""
        timeout = getattr(self.settings, "limit_order_timeout_seconds", 300)
        poll_interval = 30
        elapsed = 0
        symbol = outcome.decision.symbol
        if not outcome.order_id:
            logger.warning(
                "_await_limit_fill called with empty order_id — returning outcome unchanged"
            )
            return outcome
        order_id_int = int(outcome.order_id)

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                status_data = client.get_order(symbol, order_id_int)
                status = status_data.get("status", "")
                if status == "FILLED":
                    filled_price = self._extract_price(status_data)
                    logger.info(
                        "LIMIT order %s filled @ %s", order_id_int, filled_price
                    )
                    return TradeOutcome(
                        decision=outcome.decision,
                        order_id=outcome.order_id,
                        executed_price=filled_price,
                        pnl_usdt=None,
                        dry_run=False,
                        timestamp=outcome.timestamp,
                    )
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    logger.warning(
                        "LIMIT order %s ended with status %s", order_id_int, status
                    )
                    return TradeOutcome(
                        decision=outcome.decision.__class__(
                            symbol=outcome.decision.symbol,
                            action=Action.HOLD,
                            reasoning=f"limit_order_{status.lower()}",
                        ),
                        order_id=outcome.order_id,
                        dry_run=False,
                        timestamp=outcome.timestamp,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error polling LIMIT order %s: %s", order_id_int, exc)

        # Timeout — cancel the order
        logger.warning(
            "LIMIT order %s timed out after %ss — cancelling", order_id_int, timeout
        )
        try:
            client.cancel_order(symbol, order_id_int)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to cancel timed-out LIMIT order %s: %s", order_id_int, exc
            )
        return TradeOutcome(
            decision=outcome.decision.__class__(
                symbol=outcome.decision.symbol,
                action=Action.HOLD,
                reasoning="limit_order_timeout",
            ),
            order_id=outcome.order_id,
            dry_run=False,
            timestamp=outcome.timestamp,
        )

    async def _ensure_api_client(self) -> Any:
        async with self._lock:
            if self._api_client is not None:
                return self._api_client
            self._api_client = self._api_client_factory()
            return self._api_client

    @staticmethod
    def _extract_price(data: dict[str, Any]) -> float | None:
        # Binance MARKET orders return "price": "0.00000000" — use numeric > 0 test.
        for key in ("price", "avgPrice"):
            raw = data.get(key)
            if raw is not None:
                try:
                    val = float(raw)
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass
        for fill in data.get("fills") or []:
            raw = fill.get("price")
            if raw is not None:
                try:
                    val = float(raw)
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass
        return None

    async def _format_quantity(self, symbol: str, quantity: float, client: Any) -> str:
        """Round quantity down to the exchange's LOT_SIZE stepSize and return as string."""
        step_size = await self._get_step_size(symbol, client)
        if not step_size:
            # Step size unknown — raise so the order is blocked rather than sent
            # with wrong precision (which would cause Binance -1111).
            raise RuntimeError(
                f"Cannot determine LOT_SIZE stepSize for {symbol} — refusing to place order"
            )
        step = Decimal(step_size)
        qty = Decimal(str(quantity))
        # Floor to the nearest step using exact Decimal arithmetic (avoids float
        # imprecision, e.g. 0.07/0.01 == 6.9999... in IEEE 754).
        rounded = (qty // step) * step
        # Number of decimal places == abs(exponent of normalised step)
        decimals = max(0, -step.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"

    async def _get_step_size(self, symbol: str, client: Any) -> str:
        """Fetch and cache the LOT_SIZE stepSize for a symbol."""
        if symbol in self._step_size_cache:
            return self._step_size_cache[symbol]
        for attempt in range(3):
            try:
                info = await asyncio.to_thread(client.get_exchange_info, symbol)
                symbols = info.get("symbols") or []
                filters = symbols[0].get("filters", []) if symbols else []
                for f in filters:
                    if f.get("filterType") == "LOT_SIZE":
                        step = f.get("stepSize", "")
                        self._step_size_cache[symbol] = step
                        logger.debug("Cached stepSize=%s for %s", step, symbol)
                        return step
                break  # got a response but no LOT_SIZE filter — don't retry
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(1.0)
                else:
                    logger.warning(
                        "Could not fetch stepSize for %s after 3 attempts: %s",
                        symbol,
                        exc,
                    )
        return ""

    async def _prewarm_step_sizes(self, client: Any) -> None:
        """Fetch and cache LOT_SIZE stepSize for all configured symbols at startup."""
        symbols = getattr(self.settings, "trading_symbols", [])
        for symbol in symbols:
            step = await self._get_step_size(symbol, client)
            if step:
                logger.info("Pre-warmed stepSize=%s for %s", step, symbol)
            else:
                logger.warning("Could not pre-warm stepSize for %s — orders will be blocked", symbol)

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
