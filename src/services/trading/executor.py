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
        self._step_size_cache: dict[str, str] = {}
        self._market_step_size_cache: dict[str, str] = {}
        self._tick_size_cache: dict[str, str] = {}

    async def initialize(self) -> None:
        """Apply leverage and pre-warm exchange filters.

        Called automatically by TradingLoop.run() at startup.  Also called
        by ``__aenter__`` for callers that use the async-context-manager API.
        """
        client = await self._ensure_api_client()
        await self._apply_leverage(client)
        await self._prewarm_filters(client)

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
            decision.symbol, decision.quantity, client,
            order_type=decision.order_type,
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
        # Map close actions to the Binance side that closes the position
        _CLOSE_SIDE = {
            Action.CLOSE_LONG: "SELL",
            Action.CLOSE_SHORT: "BUY",
        }
        side = _CLOSE_SIDE.get(decision.action, decision.action.value)
        is_close = decision.action in _CLOSE_SIDE
        params = {
            "symbol": decision.symbol,
            "side": side,
            "type": decision.order_type,
            "quantity": qty_str,
        }
        if is_close and getattr(self.settings, "binance_product", "spot") == "usdt_futures":
            params["reduceOnly"] = "true"
        if decision.price is not None:
            params["price"] = await self._format_price(
                decision.symbol, decision.price, client
            )
        logger.info(
            "Submitting order: %s (stepSize=%s, marketStepSize=%s, tickSize=%s)",
            params,
            self._step_size_cache.get(decision.symbol, "?"),
            self._market_step_size_cache.get(decision.symbol, "?"),
            self._tick_size_cache.get(decision.symbol, "?"),
        )
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

        sl_stop_str = await self._format_price(symbol, sl_price, client)
        tp_stop_str = await self._format_price(symbol, tp_price, client)

        if is_futures:
            # --- Futures: try STOP_MARKET/TAKE_PROFIT_MARKET first (mainnet).
            # Fall back to STOP/TAKE_PROFIT with reduceOnly on -4120 (demo). ---
            sl_order_id = await self._place_futures_bracket(
                client, symbol, close_side, "STOP_MARKET", "STOP",
                sl_price, sl_stop_str, quantity, "SL",
            )
            tp_order_id = await self._place_futures_bracket(
                client, symbol, close_side, "TAKE_PROFIT_MARKET", "TAKE_PROFIT",
                tp_price, tp_stop_str, quantity, "TP",
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
            qty_str = await self._format_quantity(symbol, quantity, client)

            # SL: sell limit placed below trigger so it has room to fill on the way down
            if close_side == "SELL":
                sl_limit_price = sl_price * (1.0 - sl_offset)
                tp_limit_price = tp_price * (1.0 - tp_offset)
            else:
                # Closing a short: BUY limit placed above trigger
                sl_limit_price = sl_price * (1.0 + sl_offset)
                tp_limit_price = tp_price * (1.0 + tp_offset)

            sl_limit_str = await self._format_price(symbol, sl_limit_price, client)
            tp_limit_str = await self._format_price(symbol, tp_limit_price, client)

            try:
                sl_data = client.create_order(
                    symbol=symbol,
                    side=close_side,
                    type="STOP_LOSS_LIMIT",
                    quantity=qty_str,
                    stopPrice=sl_stop_str,
                    price=sl_limit_str,
                    timeInForce="GTC",
                )
                sl_order_id = str(sl_data.get("orderId", "")) or None
                logger.info(
                    "Spot SL bracket placed: %s stopPrice=%s limitPrice=%s qty=%s order=%s",
                    symbol,
                    sl_stop_str,
                    sl_limit_str,
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
                    stopPrice=tp_stop_str,
                    price=tp_limit_str,
                    timeInForce="GTC",
                )
                tp_order_id = str(tp_data.get("orderId", "")) or None
                logger.info(
                    "Spot TP bracket placed: %s stopPrice=%s limitPrice=%s qty=%s order=%s",
                    symbol,
                    tp_stop_str,
                    tp_limit_str,
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
                if oid.startswith("algo:"):
                    strategy_id = int(oid[len("algo:"):])
                    client.cancel_algo_order(strategy_id)
                else:
                    client.cancel_order(symbol, int(oid))
                logger.info("Cancelled %s bracket order %s for %s", label, oid, symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not cancel %s bracket %s for %s: %s", label, oid, symbol, exc
                )

    async def _place_futures_bracket(
        self,
        client: Any,
        symbol: str,
        side: str,
        market_type: str,
        limit_type: str,
        trigger_price: float,
        stop_str: str,
        quantity: float,
        label: str,
    ) -> str | None:
        """Place a single futures bracket order (SL or TP).

        Fallback chain (each step tried only if the previous returns -4120):
          1. STOP_MARKET / TAKE_PROFIT_MARKET with ``closePosition=true``  (mainnet)
          2. STOP_MARKET / TAKE_PROFIT_MARKET with ``reduceOnly=true`` + qty  (demo-fapi)
          3. STOP / TAKE_PROFIT limit with ``reduceOnly=true`` + 0.5 % price offset
          4. Algo Order API
        """
        qty_str = await self._format_quantity(symbol, quantity, client)

        # --- Attempt 1: market type with closePosition ---
        try:
            data = client.create_order(
                symbol=symbol,
                side=side,
                type=market_type,
                stopPrice=stop_str,
                closePosition="true",
            )
            oid = str(data.get("orderId", "")) or None
            logger.info(
                "Futures %s bracket placed: %s type=%s stopPrice=%s order=%s",
                label, symbol, market_type, stop_str, oid,
            )
            return oid
        except Exception as exc:  # noqa: BLE001
            if "-4120" not in str(exc):
                logger.error("Failed to place futures %s bracket for %s: %s", label, symbol, exc)
                return None
            logger.info(
                "%s not supported on this endpoint for %s — falling back to %s with reduceOnly",
                market_type, symbol, market_type,
            )

        # --- Attempt 2: market type with reduceOnly + quantity (demo-fapi) ---
        try:
            data = client.create_order(
                symbol=symbol,
                side=side,
                type=market_type,
                stopPrice=stop_str,
                quantity=qty_str,
                reduceOnly="true",
            )
            oid = str(data.get("orderId", "")) or None
            logger.info(
                "Futures %s bracket placed: %s type=%s stopPrice=%s qty=%s order=%s",
                label, symbol, market_type, stop_str, qty_str, oid,
            )
            return oid
        except Exception as exc:  # noqa: BLE001
            if "-4120" not in str(exc):
                logger.error("Failed to place futures %s bracket (reduceOnly) for %s: %s", label, symbol, exc)
                return None
            logger.info(
                "%s reduceOnly not supported on this endpoint for %s — falling back to %s",
                market_type, symbol, limit_type,
            )

        # --- Attempt 3: STOP / TAKE_PROFIT with reduceOnly + limit price ---
        try:
            offset = 0.005  # 0.5 % worse than trigger for fill safety
            if side == "SELL":
                limit_price = trigger_price * (1 - offset)
            else:
                limit_price = trigger_price * (1 + offset)
            limit_str = await self._format_price(symbol, limit_price, client)
            data = client.create_order(
                symbol=symbol,
                side=side,
                type=limit_type,
                stopPrice=stop_str,
                price=limit_str,
                quantity=qty_str,
                reduceOnly="true",
                timeInForce="GTC",
            )
            oid = str(data.get("orderId", "")) or None
            logger.info(
                "Futures %s bracket placed: %s type=%s stopPrice=%s price=%s qty=%s order=%s",
                label, symbol, limit_type, stop_str, limit_str, qty_str, oid,
            )
            return oid
        except Exception as exc:  # noqa: BLE001
            if "-4120" not in str(exc):
                logger.error("Failed to place futures %s bracket (%s fallback) for %s: %s", label, limit_type, symbol, exc)
                return None
            logger.info(
                "%s not supported on this endpoint for %s — falling back to Algo Order API",
                limit_type, symbol,
            )

        # --- Attempt 4: Algo Order API ---
        try:
            if label == "SL":
                data = client.create_algo_sl_order(
                    symbol=symbol, side=side, stop_price=stop_str, close_position=True,
                )
            else:
                data = client.create_algo_tp_order(
                    symbol=symbol, side=side, stop_price=stop_str, close_position=True,
                )
            strategy_id = str(data.get("strategyId", "")) or None
            oid = f"algo:{strategy_id}" if strategy_id else None
            logger.info(
                "Futures %s bracket placed via Algo API: %s stopPrice=%s strategyId=%s",
                label, symbol, stop_str, strategy_id,
            )
            return oid
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to place futures %s bracket (algo fallback) for %s: %s", label, symbol, exc)
            return None

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
        # Futures: compute from cumQuote / executedQty when price fields are zero
        cum_quote = data.get("cumQuote") or data.get("cumBase")
        exec_qty = data.get("executedQty")
        if cum_quote is not None and exec_qty is not None:
            try:
                cq, eq = float(cum_quote), float(exec_qty)
                if cq > 0 and eq > 0:
                    return cq / eq
            except (ValueError, TypeError, ZeroDivisionError):
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

    async def _format_quantity(
        self, symbol: str, quantity: float, client: Any, *, order_type: str = "LIMIT"
    ) -> str:
        """Round quantity down to the exchange's stepSize and return as string.

        MARKET orders use MARKET_LOT_SIZE (coarser on some futures pairs);
        all other order types use LOT_SIZE.
        """
        if order_type == "MARKET":
            step_size = await self._get_market_step_size(symbol, client)
        else:
            step_size = await self._get_step_size(symbol, client)
        if not step_size:
            raise RuntimeError(
                f"Cannot determine stepSize for {symbol} ({order_type}) — refusing to place order"
            )
        step = Decimal(step_size)
        qty = Decimal(str(quantity))
        # Floor to the nearest step using exact Decimal arithmetic (avoids float
        # imprecision, e.g. 0.07/0.01 == 6.9999... in IEEE 754).
        rounded = (qty // step) * step
        # Number of decimal places == abs(exponent of normalised step)
        decimals = max(0, -step.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"

    async def _format_price(self, symbol: str, price: float, client: Any) -> str:
        """Round price to the exchange's PRICE_FILTER tickSize and return as string."""
        tick_size = await self._get_tick_size(symbol, client)
        if not tick_size:
            raise RuntimeError(
                f"Cannot determine PRICE_FILTER tickSize for {symbol} — refusing to place order"
            )
        tick = Decimal(tick_size)
        p = Decimal(str(price))
        rounded = (p // tick) * tick
        decimals = max(0, -tick.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"

    async def _cache_filters(self, symbol: str, client: Any) -> None:
        """Fetch and cache LOT_SIZE, MARKET_LOT_SIZE, and PRICE_FILTER."""
        if (
            symbol in self._step_size_cache
            and symbol in self._tick_size_cache
            and symbol in self._market_step_size_cache
        ):
            return
        for attempt in range(3):
            try:
                info = await asyncio.to_thread(client.get_exchange_info, symbol)
                symbols = info.get("symbols") or []
                # Find the exact symbol — some endpoints ignore the symbol
                # query param and return all symbols.
                sym_info = next(
                    (s for s in symbols if s.get("symbol") == symbol),
                    symbols[0] if symbols else {},
                )
                filters = sym_info.get("filters", [])
                for f in filters:
                    ft = f.get("filterType")
                    if ft == "LOT_SIZE":
                        self._step_size_cache.setdefault(symbol, f.get("stepSize", ""))
                    elif ft == "MARKET_LOT_SIZE":
                        self._market_step_size_cache.setdefault(symbol, f.get("stepSize", ""))
                    elif ft == "PRICE_FILTER":
                        self._tick_size_cache.setdefault(symbol, f.get("tickSize", ""))
                # Futures may not have MARKET_LOT_SIZE — fall back to LOT_SIZE
                if symbol not in self._market_step_size_cache:
                    self._market_step_size_cache[symbol] = self._step_size_cache.get(symbol, "")
                logger.debug(
                    "Cached filters for %s: stepSize=%s marketStepSize=%s tickSize=%s",
                    symbol,
                    self._step_size_cache.get(symbol, "?"),
                    self._market_step_size_cache.get(symbol, "?"),
                    self._tick_size_cache.get(symbol, "?"),
                )
                return
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(1.0)
                else:
                    logger.warning(
                        "Could not fetch filters for %s after 3 attempts: %s",
                        symbol,
                        exc,
                    )

    async def _get_step_size(self, symbol: str, client: Any) -> str:
        """Return cached LOT_SIZE stepSize for a symbol."""
        await self._cache_filters(symbol, client)
        return self._step_size_cache.get(symbol, "")

    async def _get_market_step_size(self, symbol: str, client: Any) -> str:
        """Return cached MARKET_LOT_SIZE stepSize (falls back to LOT_SIZE)."""
        await self._cache_filters(symbol, client)
        return self._market_step_size_cache.get(symbol, "") or self._step_size_cache.get(symbol, "")

    async def _get_tick_size(self, symbol: str, client: Any) -> str:
        """Return cached PRICE_FILTER tickSize for a symbol."""
        await self._cache_filters(symbol, client)
        return self._tick_size_cache.get(symbol, "")

    async def _prewarm_filters(self, client: Any) -> None:
        """Fetch and cache LOT_SIZE, MARKET_LOT_SIZE, and PRICE_FILTER for all configured symbols."""
        symbols = getattr(self.settings, "trading_symbols", [])
        for symbol in symbols:
            await self._cache_filters(symbol, client)
            step = self._step_size_cache.get(symbol, "")
            mkt_step = self._market_step_size_cache.get(symbol, "")
            tick = self._tick_size_cache.get(symbol, "")
            if step and tick:
                logger.info(
                    "Pre-warmed filters for %s: stepSize=%s marketStepSize=%s tickSize=%s",
                    symbol, step, mkt_step, tick,
                )
            else:
                logger.warning(
                    "Could not pre-warm filters for %s (step=%s mktStep=%s tick=%s)",
                    symbol, step, mkt_step, tick,
                )

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
