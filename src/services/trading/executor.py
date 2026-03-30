"""Trade execution through the local Binance REST client."""

from __future__ import annotations

import asyncio
import logging
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
        self._api_client_factory = api_client_factory or self._default_api_client_factory
        self._api_client: Any | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "Executor":
        await self._ensure_api_client()
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
        params = {
            "symbol": decision.symbol,
            "side": decision.action.value,
            "type": decision.order_type,
            "quantity": decision.quantity,
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
    ) -> tuple[str | None, str | None]:
        """Place STOP_MARKET (SL) and TAKE_PROFIT_MARKET (TP) reduce-only orders.

        Both use ``closePosition=true`` so Binance closes the entire position when
        triggered.  Returns ``(sl_order_id, tp_order_id)``; either may be None if
        the individual order fails (entry position is still open — the error is
        logged and the bot will attempt a bot-managed exit on the next reconcile).
        """
        client = await self._ensure_api_client()
        close_side = "SELL" if entry_side == "BUY" else "BUY"
        sl_order_id: str | None = None
        tp_order_id: str | None = None

        try:
            sl_data = client.create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=f"{sl_price:.2f}",
                closePosition="true",
            )
            sl_order_id = str(sl_data.get("orderId", "")) or None
            logger.info("SL bracket placed: %s stopPrice=%.2f order=%s", symbol, sl_price, sl_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to place SL bracket for %s: %s", symbol, exc)

        try:
            tp_data = client.create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=f"{tp_price:.2f}",
                closePosition="true",
            )
            tp_order_id = str(tp_data.get("orderId", "")) or None
            logger.info("TP bracket placed: %s stopPrice=%.2f order=%s", symbol, tp_price, tp_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to place TP bracket for %s: %s", symbol, exc)

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
                logger.warning("Could not cancel %s bracket %s for %s: %s", label, oid, symbol, exc)

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

    async def _await_limit_fill(self, outcome: TradeOutcome, client: Any) -> TradeOutcome:
        """Poll until a LIMIT order is filled or the timeout elapses, then cancel."""
        timeout = getattr(self.settings, "limit_order_timeout_seconds", 300)
        poll_interval = 30
        elapsed = 0
        symbol = outcome.decision.symbol
        if not outcome.order_id:
            logger.warning("_await_limit_fill called with empty order_id — returning outcome unchanged")
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
                    logger.info("LIMIT order %s filled @ %s", order_id_int, filled_price)
                    return TradeOutcome(
                        decision=outcome.decision,
                        order_id=outcome.order_id,
                        executed_price=filled_price,
                        pnl_usdt=None,
                        dry_run=False,
                        timestamp=outcome.timestamp,
                    )
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    logger.warning("LIMIT order %s ended with status %s", order_id_int, status)
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
        logger.warning("LIMIT order %s timed out after %ss — cancelling", order_id_int, timeout)
        try:
            client.cancel_order(symbol, order_id_int)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to cancel timed-out LIMIT order %s: %s", order_id_int, exc)
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

    def _default_api_client_factory(self) -> Any:
        return BinanceRestClient(
            api_key=self.settings.binance_api_key,
            api_secret=self.settings.binance_api_secret,
            product=self.settings.binance_product,
            testnet=self.settings.binance_testnet,
            base_url=self.settings.binance_base_url,
        )
