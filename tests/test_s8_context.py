"""Acceptance tests for S8 context assembly."""

from __future__ import annotations

from src.mcp_servers.config import Settings
from src.mcp_servers.shared.domain.analysis import (
    IndicatorSet,
    PatternResult,
    Signal,
    TechnicalAnalysis,
)
from src.mcp_servers.shared.domain.market import MarketSnapshot, OHLCVCandle
from src.legacy.services.analysis import ContextBuilder, build_system_prompt


def build_settings(*, model_supports_vision: bool = False) -> Settings:
    return Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        max_order_usdt=75.0,
        model_supports_vision=model_supports_vision,
    )


def candle() -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=1, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0
    )


def snapshot(symbol: str, price: float, change: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        price=price,
        change_24h_pct=change,
        volume_24h=123456.0,
        bid=price - 1,
        ask=price + 1,
        candles=[candle()] * 60,
        timestamp=1,
    )


def analysis(symbol: str, signal: Signal, reasoning: str) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        symbol=symbol,
        signal=signal,
        indicators=IndicatorSet(
            rsi_14=35.0,
            macd_line=1.0,
            macd_signal=0.8,
            macd_hist=0.2,
            bb_upper=105.0,
            bb_mid=100.0,
            bb_lower=95.0,
            ema_20=99.0,
            ema_50=98.0,
            volume_sma_20=1000.0,
        ),
        reasoning=reasoning,
    )


def pattern(symbol: str, chart_png_b64: str = "") -> PatternResult:
    return PatternResult(
        symbol=symbol,
        patterns=["double_bottom"],
        support=95.0,
        resistance=105.0,
        chart_png_b64=chart_png_b64,
    )


def test_system_prompt_includes_max_order_and_contract() -> None:
    prompt = build_system_prompt(75.0)

    assert "autonomous crypto trading agent" in prompt
    assert "BTCUSDT, ETHUSDT only" in prompt
    assert "75.0 USDT" in prompt
    assert "MARKET, LIMIT only" in prompt
    assert "Choose the single best opportunity" in prompt
    assert "Use HOLD only when both symbols lack a clear edge" in prompt
    assert '{"action":"HOLD","symbol","reasoning"}' in prompt


def test_build_returns_message_pair_for_chat_api() -> None:
    builder = ContextBuilder(build_settings())

    system_prompt, user_message = builder.build(
        snapshots={
            "BTCUSDT": snapshot("BTCUSDT", 64000.0, 2.5),
            "ETHUSDT": snapshot("ETHUSDT", 3200.0, 1.5),
        },
        analyses={
            "BTCUSDT": analysis("BTCUSDT", Signal.BUY, "BTC is constructive."),
            "ETHUSDT": analysis("ETHUSDT", Signal.NEUTRAL, "ETH is range bound."),
        },
        patterns={
            "BTCUSDT": pattern("BTCUSDT"),
            "ETHUSDT": pattern("ETHUSDT"),
        },
        rag_context="=== RECENT NEWS (BTC/ETH) ===\n[1] Example",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "## Market Snapshot" in user_message
    assert "## Patterns Detected" in user_message
    assert "## RAG Context" in user_message
    assert "## Your Decision" in user_message


def test_user_message_is_capped_at_8000_chars() -> None:
    builder = ContextBuilder(build_settings())
    long_context = "crypto " * 3000

    _system_prompt, user_message = builder.build(
        snapshots={
            "BTCUSDT": snapshot("BTCUSDT", 64000.0, 2.5),
            "ETHUSDT": snapshot("ETHUSDT", 3200.0, 1.5),
        },
        analyses={
            "BTCUSDT": analysis("BTCUSDT", Signal.BUY, "BTC is constructive."),
            "ETHUSDT": analysis("ETHUSDT", Signal.NEUTRAL, "ETH is range bound."),
        },
        patterns={
            "BTCUSDT": pattern("BTCUSDT"),
            "ETHUSDT": pattern("ETHUSDT"),
        },
        rag_context=long_context,
    )

    assert len(user_message) <= 8000


def test_chart_base64_only_included_when_vision_enabled() -> None:
    snapshots = {
        "BTCUSDT": snapshot("BTCUSDT", 64000.0, 2.5),
        "ETHUSDT": snapshot("ETHUSDT", 3200.0, 1.5),
    }
    analyses = {
        "BTCUSDT": analysis("BTCUSDT", Signal.BUY, "BTC is constructive."),
        "ETHUSDT": analysis("ETHUSDT", Signal.NEUTRAL, "ETH is range bound."),
    }
    patterns = {
        "BTCUSDT": pattern("BTCUSDT", chart_png_b64="AAAABBBB"),
        "ETHUSDT": pattern("ETHUSDT", chart_png_b64="CCCCDDDD"),
    }

    disabled_message = ContextBuilder(
        build_settings(model_supports_vision=False)
    ).build(snapshots, analyses, patterns, "none",)[1]
    enabled_message = ContextBuilder(build_settings(model_supports_vision=True)).build(
        snapshots,
        analyses,
        patterns,
        "none",
    )[1]

    assert "AAAABBBB" not in disabled_message
    assert "CCCCDDDD" not in disabled_message
    assert "AAAABBBB" in enabled_message
    assert "CCCCDDDD" in enabled_message
