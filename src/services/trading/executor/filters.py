"""Exchange LOT_SIZE / PRICE_FILTER caching and rounding helpers.

The :class:`ExchangeFilters` holds per-symbol step/tick caches and exposes
async helpers that fetch filters on demand and round quantities and prices
to the exchange's tick grid using exact Decimal arithmetic.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class ExchangeFilters:
    """Cache LOT_SIZE, MARKET_LOT_SIZE, and PRICE_FILTER per symbol."""

    def __init__(self) -> None:
        self.step_size_cache: dict[str, str] = {}
        self.market_step_size_cache: dict[str, str] = {}
        self.tick_size_cache: dict[str, str] = {}

    async def cache(self, symbol: str, client: Any) -> None:
        """Fetch and cache LOT_SIZE, MARKET_LOT_SIZE, and PRICE_FILTER."""
        if (
            symbol in self.step_size_cache
            and symbol in self.tick_size_cache
            and symbol in self.market_step_size_cache
        ):
            return
        for attempt in range(3):
            try:
                info = await asyncio.to_thread(client.get_exchange_info, symbol)
                symbols = info.get("symbols") or []
                sym_info = next(
                    (s for s in symbols if s.get("symbol") == symbol),
                    symbols[0] if symbols else {},
                )
                filters = sym_info.get("filters", [])
                for f in filters:
                    ft = f.get("filterType")
                    if ft == "LOT_SIZE":
                        self.step_size_cache.setdefault(symbol, f.get("stepSize", ""))
                    elif ft == "MARKET_LOT_SIZE":
                        self.market_step_size_cache.setdefault(symbol, f.get("stepSize", ""))
                    elif ft == "PRICE_FILTER":
                        self.tick_size_cache.setdefault(symbol, f.get("tickSize", ""))
                if symbol not in self.market_step_size_cache:
                    self.market_step_size_cache[symbol] = self.step_size_cache.get(symbol, "")
                logger.debug(
                    "Cached filters for %s: stepSize=%s marketStepSize=%s tickSize=%s",
                    symbol,
                    self.step_size_cache.get(symbol, "?"),
                    self.market_step_size_cache.get(symbol, "?"),
                    self.tick_size_cache.get(symbol, "?"),
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

    async def get_step_size(self, symbol: str, client: Any) -> str:
        await self.cache(symbol, client)
        return self.step_size_cache.get(symbol, "")

    async def get_market_step_size(self, symbol: str, client: Any) -> str:
        await self.cache(symbol, client)
        return (
            self.market_step_size_cache.get(symbol, "")
            or self.step_size_cache.get(symbol, "")
        )

    async def get_tick_size(self, symbol: str, client: Any) -> str:
        await self.cache(symbol, client)
        return self.tick_size_cache.get(symbol, "")

    async def format_quantity(
        self,
        symbol: str,
        quantity: float,
        client: Any,
        *,
        order_type: str = "LIMIT",
    ) -> str:
        """Round quantity down to stepSize and return as string.

        MARKET orders use MARKET_LOT_SIZE (coarser on some futures pairs);
        all other order types use LOT_SIZE.
        """
        if order_type == "MARKET":
            step_size = await self.get_market_step_size(symbol, client)
        else:
            step_size = await self.get_step_size(symbol, client)
        if not step_size:
            raise RuntimeError(
                f"Cannot determine stepSize for {symbol} ({order_type}) — refusing to place order"
            )
        step = Decimal(step_size)
        qty = Decimal(str(quantity))
        # Floor to the nearest step using exact Decimal arithmetic (avoids float
        # imprecision, e.g. 0.07/0.01 == 6.9999... in IEEE 754).
        rounded = (qty // step) * step
        decimals = max(0, -step.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"

    async def format_price(self, symbol: str, price: float, client: Any) -> str:
        """Round price to PRICE_FILTER tickSize and return as string."""
        tick_size = await self.get_tick_size(symbol, client)
        if not tick_size:
            raise RuntimeError(
                f"Cannot determine PRICE_FILTER tickSize for {symbol} — refusing to place order"
            )
        tick = Decimal(tick_size)
        p = Decimal(str(price))
        rounded = (p // tick) * tick
        decimals = max(0, -tick.normalize().as_tuple().exponent)
        return f"{rounded:.{decimals}f}"

    async def prewarm(self, symbols: Iterable[str], client: Any) -> None:
        """Fetch and cache filters for all configured symbols at startup."""
        for symbol in symbols:
            await self.cache(symbol, client)
            step = self.step_size_cache.get(symbol, "")
            mkt_step = self.market_step_size_cache.get(symbol, "")
            tick = self.tick_size_cache.get(symbol, "")
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
