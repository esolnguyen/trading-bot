"""Settings parsing + startup gate behaviour."""

from __future__ import annotations

import pytest

from src.trading_bot.config import TradingBotSettings


def test_defaults(clean_env) -> None:
    s = TradingBotSettings()
    assert s.trading_symbols == ["BTCUSDT"]
    assert s.primary_timeframe == "15m"
    assert s.decision_interval_seconds == 900
    assert s.min_conviction == 6
    assert s.bot_mode == "dry_run"
    assert s.llm_model == "claude-sonnet-4-6"
    assert s.llm_max_iterations == 12
    assert s.anthropic_api_key == ""


def test_csv_trading_symbols(clean_env) -> None:
    clean_env.setenv("TRADING_SYMBOLS", "btcusdt, ethusdt ,solusdt")
    s = TradingBotSettings()
    assert s.trading_symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_bot_mode_normalized(clean_env) -> None:
    clean_env.setenv("BOT_MODE", "  DRY_RUN  ")
    s = TradingBotSettings()
    assert s.bot_mode == "dry_run"


def test_assert_runnable_refuses_live(clean_env) -> None:
    clean_env.setenv("BOT_MODE", "live")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = TradingBotSettings()
    with pytest.raises(RuntimeError, match="BOT_MODE=live"):
        s.assert_runnable()


def test_assert_runnable_requires_anthropic_key(clean_env) -> None:
    clean_env.setenv("BOT_MODE", "dry_run")
    s = TradingBotSettings()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        s.assert_runnable()


def test_assert_runnable_refuses_empty_symbols(clean_env) -> None:
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    clean_env.setenv("TRADING_SYMBOLS", "   ,  ")
    s = TradingBotSettings()
    with pytest.raises(RuntimeError, match="TRADING_SYMBOLS"):
        s.assert_runnable()


def test_assert_runnable_happy_path(clean_env) -> None:
    clean_env.setenv("BOT_MODE", "dry_run")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    clean_env.setenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT")
    TradingBotSettings().assert_runnable()
