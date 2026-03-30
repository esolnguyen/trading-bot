"""Acceptance tests for S10 risk management."""

from __future__ import annotations

import logging

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision
from src.services.trading import RiskManager


def build_settings(*, bot_enabled: bool = True, bot_dry_run: bool = False, max_order_usdt: float = 50.0) -> Settings:
    return Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        bot_enabled=bot_enabled,
        bot_dry_run=bot_dry_run,
        max_order_usdt=max_order_usdt,
    )


def buy_decision(quantity: float = 1.0, price: float | None = 10.0, order_type: str = "MARKET") -> TradeDecision:
    return TradeDecision(
        symbol="BTCUSDT",
        action=Action.BUY,
        quantity=quantity,
        order_type=order_type,
        price=price,
        reasoning="test",
        confidence=0.7,
        timestamp=1,
        source="grok",
    )


def test_bot_enabled_false_always_returns_hold(caplog) -> None:
    caplog.set_level(logging.WARNING)
    manager = RiskManager(build_settings(bot_enabled=False))

    result = manager.validate(buy_decision(), {"USDT": 1000, "prices": {"BTCUSDT": 10}})

    assert result.decision.action is Action.HOLD
    assert result.status == "blocked"
    assert "BOT_ENABLED is false" in caplog.text


def test_bot_dry_run_true_allows_decision_through() -> None:
    manager = RiskManager(build_settings(bot_enabled=True, bot_dry_run=True))

    result = manager.validate(buy_decision(quantity=1.0, price=10.0), {"USDT": 1000, "prices": {"BTCUSDT": 10}})

    assert result.decision.action is Action.BUY
    assert result.dry_run is True


def test_order_exceeding_max_is_clamped_not_rejected() -> None:
    manager = RiskManager(build_settings(bot_enabled=True, max_order_usdt=50.0))

    result = manager.validate(buy_decision(quantity=10.0, price=20.0), {"USDT": 1000, "prices": {"BTCUSDT": 20}})

    assert result.decision.action is Action.BUY
    assert round(result.decision.quantity, 8) == 2.5
    assert result.status == "clamped"


def test_insufficient_balance_returns_hold_with_reason() -> None:
    manager = RiskManager(build_settings(bot_enabled=True))

    result = manager.validate(buy_decision(quantity=5.0, price=10.0), {"USDT": 20, "prices": {"BTCUSDT": 10}})

    assert result.decision.action is Action.HOLD
    assert "insufficient_balance" in result.decision.reasoning
    assert result.status == "blocked"


def test_dry_run_and_hold_logged_at_info_or_debug(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    manager = RiskManager(build_settings(bot_enabled=True, bot_dry_run=True))

    manager.validate(TradeDecision(symbol="BTCUSDT", action=Action.HOLD), {"USDT": 1000, "prices": {"BTCUSDT": 10}})

    # dry_run and HOLD passthrough should not pollute WARNING/ERROR channels
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    # but something should be logged
    assert len(caplog.records) > 0


def test_limit_order_requires_price() -> None:
    manager = RiskManager(build_settings(bot_enabled=True))

    result = manager.validate(buy_decision(quantity=1.0, price=None, order_type="LIMIT"), {"USDT": 1000, "prices": {"BTCUSDT": 10}})

    assert result.decision.action is Action.HOLD
    assert result.decision.reasoning == "limit_price_required"
