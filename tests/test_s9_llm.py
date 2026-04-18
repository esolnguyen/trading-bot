"""Acceptance tests for S9 LLM manager."""

from __future__ import annotations

import asyncio
import json

from src.mcp_servers.config import Settings
from src.legacy.domain.trading import Action
from src.mcp_servers.shared.infrastructure.ai.llm import (
    LLMManager,
    _normalize_azure_openai_endpoint,
    _should_use_azure_foundry_anthropic_client,
)


def build_settings(*, model_supports_vision: bool = False) -> Settings:
    return Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        model_supports_vision=model_supports_vision,
    )


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str, usage=None) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeUsage:
    def __init__(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RetryableError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_valid_json_maps_to_trade_decision() -> None:
    payload = json.dumps(
        {
            "action": "BUY",
            "symbol": "BTCUSDT",
            "quantity": 0.001,
            "order_type": "MARKET",
            "reasoning": "Momentum confirmed",
            "confidence": 0.82,
            "timestamp": 1700000000,
        }
    )
    client = FakeClient([FakeResponse(payload, usage=FakeUsage(120, 35, 155))])
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.BUY
    assert decision.symbol == "BTCUSDT"
    assert decision.quantity == 0.001
    assert decision.confidence == 0.82
    assert decision.source == "azure"
    assert manager.last_usage == {
        "prompt_tokens": 120,
        "completion_tokens": 35,
        "total_tokens": 155,
    }
    assert manager.last_error is None
    assert manager.last_prompt["system_prompt"] == "system"
    assert manager.last_prompt["user_message"] == "user"
    assert manager.last_raw_response == payload


def test_malformed_json_returns_hold_fallback() -> None:
    client = FakeClient([FakeResponse("{not-json}")])
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.HOLD
    assert decision.source == "fallback_hold"
    assert decision.reasoning == "parse_error"
    assert manager.last_error.startswith("Malformed JSON")
    assert manager.last_raw_response == "{not-json}"


def test_invalid_symbol_returns_hold_fallback() -> None:
    payload = json.dumps(
        {
            "action": "BUY",
            "symbol": "DOGEUSDT",
            "quantity": 1,
            "order_type": "MARKET",
            "reasoning": "bad symbol",
        }
    )
    client = FakeClient([FakeResponse(payload)])
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.HOLD
    assert decision.reasoning == "parse_error"
    assert manager.last_error == "Invalid symbol"


def test_minimal_hold_payload_is_accepted_with_safe_defaults() -> None:
    client = FakeClient([FakeResponse('{"action":"HOLD"}')])
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.HOLD
    assert decision.symbol == "BTCUSDT"
    assert decision.quantity == 0.0
    assert decision.order_type == "MARKET"
    assert decision.reasoning == "model_hold_without_reason"
    assert decision.source == "azure"
    assert manager.last_error is None


def test_hold_payload_keeps_optional_reasoning_and_symbol_when_valid() -> None:
    payload = json.dumps(
        {
            "action": "HOLD",
            "symbol": "ETHUSDT",
            "reasoning": "No clear edge",
            "confidence": 0.2,
            "timestamp": 1700000000,
        }
    )
    client = FakeClient([FakeResponse(payload)])
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.HOLD
    assert decision.symbol == "ETHUSDT"
    assert decision.reasoning == "No clear edge"
    assert decision.confidence == 0.2
    assert decision.timestamp == 1700000000


def test_http_429_retries_twice_then_returns_hold() -> None:
    sleep_calls = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    client = FakeClient([RetryableError(429), RetryableError(429), RetryableError(429)])
    manager = LLMManager(
        build_settings(), client_factory=lambda _settings: client, sleep_func=fake_sleep
    )

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.HOLD
    assert decision.source == "fallback_hold"
    assert sleep_calls == [2, 4]
    assert len(client.calls) == 3
    assert manager.last_error == "status 429"


def test_vision_message_includes_image_only_when_enabled() -> None:
    payload = json.dumps(
        {
            "action": "HOLD",
            "symbol": "BTCUSDT",
            "quantity": 0,
            "order_type": "MARKET",
            "reasoning": "uncertain",
        }
    )
    enabled_client = FakeClient([FakeResponse(payload)])
    disabled_client = FakeClient([FakeResponse(payload)])

    enabled = LLMManager(
        build_settings(model_supports_vision=True),
        client_factory=lambda _settings: enabled_client,
    )
    disabled = LLMManager(
        build_settings(model_supports_vision=False),
        client_factory=lambda _settings: disabled_client,
    )

    asyncio.run(enabled.decide("system", "user", chart_b64="AAAABBBB"))
    asyncio.run(disabled.decide("system", "user", chart_b64="AAAABBBB"))

    enabled_content = enabled_client.calls[0]["messages"][1]["content"]
    disabled_content = disabled_client.calls[0]["messages"][1]["content"]

    assert isinstance(enabled_content, list)
    assert enabled_content[1]["type"] == "image_url"
    assert disabled_content == "user"


def test_unsupported_response_format_retries_without_it() -> None:
    payload = json.dumps(
        {
            "action": "BUY",
            "symbol": "BTCUSDT",
            "quantity": 0.001,
            "order_type": "MARKET",
            "reasoning": "Momentum confirmed",
        }
    )
    client = FakeClient(
        [
            RuntimeError("Unsupported `response_format` {'type': 'json_object'}"),
            FakeResponse(payload),
        ]
    )
    manager = LLMManager(build_settings(), client_factory=lambda _settings: client)

    decision = asyncio.run(manager.decide("system", "user"))

    assert decision.action is Action.BUY
    assert len(client.calls) == 2
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]


def test_normalize_azure_openai_endpoint_strips_chat_completions_suffix() -> None:
    endpoint = "https://thang-sw-central.services.ai.azure.com/models/chat/completions"

    normalized = _normalize_azure_openai_endpoint(endpoint)

    assert normalized == "https://thang-sw-central.services.ai.azure.com"


def test_foundry_endpoint_only_uses_anthropic_for_claude_deployments() -> None:
    endpoint = "https://thang-sw-central.services.ai.azure.com/models"

    assert (
        _should_use_azure_foundry_anthropic_client(endpoint, "claude-3-7-sonnet")
        is True
    )
    assert _should_use_azure_foundry_anthropic_client(endpoint, "grok-3-mini") is False
    assert (
        _should_use_azure_foundry_anthropic_client(
            "https://thang-sw-central.services.ai.azure.com/anthropic/v1/messages",
            "custom-deployment",
        )
        is True
    )
    assert (
        _should_use_azure_foundry_anthropic_client(
            "https://example-resource.openai.azure.com",
            "claude-3-7-sonnet",
        )
        is False
    )
