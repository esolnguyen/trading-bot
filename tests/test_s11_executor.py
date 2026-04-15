"""Acceptance tests for S11 trade execution."""

from __future__ import annotations

import asyncio

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision
from src.services.trading import Executor


def build_settings(**overrides) -> Settings:
    base = dict(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        bot_mode="live",
    )
    base.update(overrides)
    return Settings(**base)


def futures_settings(**overrides) -> Settings:
    return build_settings(binance_product="usdt_futures", **overrides)


def buy_decision() -> TradeDecision:
    return TradeDecision(
        symbol="BTCUSDT",
        action=Action.BUY,
        quantity=0.001,
        order_type="MARKET",
        price=None,
        reasoning="momentum",
        confidence=0.8,
        timestamp=1700000000,
        source="grok",
    )


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.cancel_calls = []
        self.closed = False
        self._positions: list[dict] = []

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def cancel_order(self, symbol: str, order_id: int):
        self.cancel_calls.append((symbol, order_id))
        return {}

    def get_open_positions(self, symbol: str):
        return self._positions

    def get_exchange_info(self, symbol: str):
        return {
            "symbols": [{
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.00100000"},
                    {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.00100000"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                ]
            }]
        }

    def close_connection(self):
        self.closed = True


def test_dry_run_never_calls_create_order() -> None:
    client = FakeClient({})
    executor = Executor(build_settings(), api_client_factory=lambda: client)

    outcome = asyncio.run(executor.execute(buy_decision(), dry_run=True))
    asyncio.run(executor.close())

    assert outcome.order_id is None
    assert outcome.dry_run is True
    assert client.calls == []


def test_hold_decision_returns_clean_outcome() -> None:
    client = FakeClient({})
    executor = Executor(build_settings(), api_client_factory=lambda: client)
    decision = TradeDecision(symbol="BTCUSDT", action=Action.HOLD, timestamp=1700000000)

    outcome = asyncio.run(executor.execute(decision, dry_run=False))
    asyncio.run(executor.close())

    assert outcome.order_id is None
    assert outcome.executed_price is None
    assert client.calls == []


def test_successful_market_order_returns_outcome_with_order_id() -> None:
    client = FakeClient(
        {
            "orderId": 12345,
            "price": "0",
            "fills": [{"price": "64250.5"}],
            "transactTime": 1700000001,
        }
    )
    executor = Executor(build_settings(), api_client_factory=lambda: client)

    outcome = asyncio.run(executor.execute(buy_decision(), dry_run=False))
    asyncio.run(executor.close())

    assert outcome.order_id == "12345"
    assert outcome.executed_price == 64250.5
    assert outcome.dry_run is False
    assert client.calls[0]["symbol"] == "BTCUSDT"
    assert client.closed is True


def test_api_error_raises_runtime_error() -> None:
    client = FakeClient(RuntimeError("order rejected"))
    executor = Executor(build_settings(), api_client_factory=lambda: client)

    try:
        asyncio.run(executor.execute(buy_decision(), dry_run=False))
    except RuntimeError as exc:
        assert str(exc) == "order rejected"
    else:
        raise AssertionError("Expected RuntimeError for API error response")
    finally:
        asyncio.run(executor.close())


# ---------------------------------------------------------------------------
# Bracket order tests
# ---------------------------------------------------------------------------

def test_place_bracket_orders_long_entry_correct_params() -> None:
    """BUY entry → close side is SELL; SL below, TP above entry price."""
    client = FakeClient({"orderId": 999})
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    sl_id, tp_id = asyncio.run(
        executor.place_bracket_orders("ETHUSDT", "BUY", sl_price=2000.0, tp_price=2200.0)
    )
    asyncio.run(executor.close())

    assert sl_id == "999"
    assert tp_id == "999"

    types = {c["type"] for c in client.calls}
    assert types == {"STOP_MARKET", "TAKE_PROFIT_MARKET"}

    for call in client.calls:
        assert call["side"] == "SELL"
        assert call["closePosition"] == "true"
        assert call["symbol"] == "ETHUSDT"


def test_place_bracket_orders_short_entry_correct_params() -> None:
    """SELL entry → close side is BUY."""
    client = FakeClient({"orderId": 888})
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    asyncio.run(
        executor.place_bracket_orders("ETHUSDT", "SELL", sl_price=2100.0, tp_price=1900.0)
    )
    asyncio.run(executor.close())

    for call in client.calls:
        assert call["side"] == "BUY"


def test_place_bracket_orders_returns_none_on_api_failure() -> None:
    """Bracket placement failure must not raise — returns None IDs."""
    client = FakeClient(RuntimeError("margin insufficient"))
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    sl_id, tp_id = asyncio.run(
        executor.place_bracket_orders("ETHUSDT", "BUY", sl_price=2000.0, tp_price=2200.0)
    )
    asyncio.run(executor.close())

    assert sl_id is None
    assert tp_id is None


def test_cancel_bracket_orders_cancels_both() -> None:
    client = FakeClient({})
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    asyncio.run(executor.cancel_bracket_orders("ETHUSDT", "111", "222"))
    asyncio.run(executor.close())

    cancelled_ids = {oid for _, oid in client.cancel_calls}
    assert cancelled_ids == {111, 222}


def test_cancel_bracket_orders_skips_when_both_none() -> None:
    client = FakeClient({})
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    asyncio.run(executor.cancel_bracket_orders("ETHUSDT", None, None))
    asyncio.run(executor.close())

    assert client.cancel_calls == []


def test_get_live_position_size_sums_amounts() -> None:
    client = FakeClient({})
    client._positions = [
        {"positionAmt": "0.024"},
        {"positionAmt": "-0.001"},  # opposite side (rare) — abs summed
    ]
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    size = asyncio.run(executor.get_live_position_size("ETHUSDT"))
    asyncio.run(executor.close())

    assert abs(size - 0.025) < 1e-9


def test_get_live_position_size_returns_inf_on_error() -> None:
    """API failure must return inf so a transient error never clears local state."""
    client = FakeClient({})
    client._positions = RuntimeError("network error")

    def raise_on_call(symbol):
        raise RuntimeError("network error")

    client.get_open_positions = raise_on_call
    executor = Executor(futures_settings(), api_client_factory=lambda: client)

    size = asyncio.run(executor.get_live_position_size("ETHUSDT"))
    asyncio.run(executor.close())

    assert size == float("inf")


# ---------------------------------------------------------------------------
# Close action side mapping tests
# ---------------------------------------------------------------------------

def test_close_long_sends_sell_side() -> None:
    """CLOSE_LONG must send side=SELL to Binance, not 'CLOSE_LONG'."""
    client = FakeClient(
        {"orderId": 55555, "avgPrice": "64300.0", "transactTime": 1700000002}
    )
    executor = Executor(build_settings(), api_client_factory=lambda: client)
    decision = TradeDecision(
        symbol="BTCUSDT", action=Action.CLOSE_LONG,
        quantity=0.001, order_type="MARKET", timestamp=1700000000,
    )

    outcome = asyncio.run(executor.execute(decision, dry_run=False))
    asyncio.run(executor.close())

    assert outcome.order_id == "55555"
    assert client.calls[0]["side"] == "SELL"


def test_close_short_sends_buy_side_with_reduce_only() -> None:
    """CLOSE_SHORT on futures must send side=BUY + reduceOnly."""
    client = FakeClient(
        {"orderId": 66666, "avgPrice": "1800.0", "transactTime": 1700000003}
    )
    executor = Executor(futures_settings(), api_client_factory=lambda: client)
    decision = TradeDecision(
        symbol="ETHUSDT", action=Action.CLOSE_SHORT,
        quantity=0.5, order_type="MARKET", timestamp=1700000000,
    )

    outcome = asyncio.run(executor.execute(decision, dry_run=False))
    asyncio.run(executor.close())

    assert outcome.order_id == "66666"
    assert client.calls[0]["side"] == "BUY"
    assert client.calls[0]["reduceOnly"] == "true"
