"""Acceptance tests for S1 settings."""

from __future__ import annotations

import os

import pytest

from src.core.config import Settings


REQUIRED_ENV = {
    "AZURE_ENDPOINT": "https://example-resource.openai.azure.com",
    "AZURE_API_KEY": "azure-key-1234",
    "AZURE_DEPLOYMENT": "grok-prod",
    "BINANCE_API_KEY": "binance-key-1234",
    "BINANCE_API_SECRET": "binance-secret-1234",
    "CRYPTOCOMPARE_API_KEY": "cc-key-1234",
}

MISSING_ENV_FILE = "/tmp/codex-nonexistent.env"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "AZURE_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_DEPLOYMENT",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_PRODUCT",
        "BINANCE_BASE_URL",
        "BINANCE_TESTNET",
        "BOT_INTERVAL_SECONDS",
        "MAX_ORDER_USDT",
        "BOT_MODE",
        "BOT_ENABLED",
        "BOT_DRY_RUN",
        "TRADING_ENGINE",
        "TRADER_SKILLS",
        "LLM_TRADER_SKILLS",
        "USE_SIGNAL_SCORER",
        "CRYPTOCOMPARE_API_KEY",
        "CHROMA_PATH",
        "NEWS_INTERVAL",
        "MACRO_INTERVAL",
        "OHLCV_INTERVAL",
    ):
        monkeypatch.delenv(key, raising=False)


def _apply_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


def test_missing_required_field_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.delenv("AZURE_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="AZURE_ENDPOINT"):
        Settings.from_env(env_file=MISSING_ENV_FILE)


def test_safe_defaults_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)

    settings = Settings.from_env(env_file=MISSING_ENV_FILE)

    assert settings.binance_testnet is True
    assert settings.binance_product == "spot"
    assert settings.binance_base_url == ""
    assert settings.bot_dry_run is True
    assert settings.bot_enabled is False


def test_settings_are_typed_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "600")
    monkeypatch.setenv("MAX_ORDER_USDT", "125.5")
    monkeypatch.setenv("NEWS_INTERVAL", "1200")

    settings = Settings.from_env(env_file=MISSING_ENV_FILE)

    assert settings.bot_interval_seconds == 600
    assert isinstance(settings.bot_interval_seconds, int)
    assert settings.max_order_usdt == 125.5
    assert isinstance(settings.max_order_usdt, float)
    assert settings.news_interval == 1200
    assert isinstance(settings.news_interval, int)


def test_max_order_must_be_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.setenv("MAX_ORDER_USDT", "10001")

    with pytest.raises(ValueError, match="max_order_usdt"):
        Settings.from_env(env_file=MISSING_ENV_FILE)


def test_bot_interval_must_not_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "-1")

    with pytest.raises(ValueError, match="bot_interval_seconds"):
        Settings.from_env(env_file=MISSING_ENV_FILE)


def test_repr_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)

    rendered = repr(Settings.from_env(env_file=MISSING_ENV_FILE))

    assert "azure-key-1234" not in rendered
    assert "binance-secret-1234" not in rendered
    assert "cc-key-1234" not in rendered


def test_binance_product_and_base_url_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.setenv("BINANCE_PRODUCT", "usdt_futures")
    monkeypatch.setenv("BINANCE_BASE_URL", "https://demo-fapi.binance.com")

    settings = Settings.from_env(env_file=MISSING_ENV_FILE)

    assert settings.binance_product == "usdt_futures"
    assert settings.binance_base_url == "https://demo-fapi.binance.com"


def test_invalid_binance_product_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_required_env(monkeypatch)
    monkeypatch.setenv("BINANCE_PRODUCT", "margin")

    with pytest.raises(ValueError, match="binance_product"):
        Settings.from_env(env_file=MISSING_ENV_FILE)
