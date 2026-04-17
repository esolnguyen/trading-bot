"""Multi-provider LLM manager with safe parsing and fallback behavior.

The heavy lifting lives in sibling modules:
  * :mod:`llm_parsing`         — error types, parsers, LLMCallResult, helpers
  * :mod:`llm_azure`           — Azure OpenAI + Azure AI Foundry (Anthropic)
  * :mod:`llm_multi_provider`  — Google / OpenRouter / LM Studio / BlockRun
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Callable, Dict, List, Union

from src.core.config import Settings
from src.domain.trading import TradeDecision
from .azure import (
    _AnthropicResponseAdapter,
    _normalize_azure_openai_endpoint,
    _should_use_azure_foundry_anthropic_client,
    complete_azure,
    decide_azure,
    default_azure_client_factory,
    extract_content,
)
from .multi_provider import (
    build_orchestrator,
    build_unified_parser,
    close_orchestrator,
    decide_multi_provider,
)
from .parsing import (
    MAX_LOG_TEXT_CHARS,
    LLMCallResult,
    ParseError,
    ResponseShapeError,
    truncate_for_log,
)

logger = logging.getLogger(__name__)


class LLMManager:
    """Send prompt context to the model and parse a safe trade decision.

    Supports Azure OpenAI (legacy path) and the full multi-provider chain
    via ProviderOrchestrator (Google, OpenRouter, LM Studio, BlockRun).
    """

    MAX_LOG_TEXT_CHARS = MAX_LOG_TEXT_CHARS

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[[Settings], Any] | None = None,
        sleep_func: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        _symbols_list: list[str] = list(
            getattr(settings, "trading_symbols", None) or ["BTCUSDT", "ETHUSDT"]
        )
        self.ALLOWED_SYMBOLS: frozenset[str] = frozenset(_symbols_list)
        self._primary_symbol: str = _symbols_list[0]
        self._sleep_func = sleep_func
        self._client_factory = client_factory or default_azure_client_factory
        self._client: Any | None = None
        self._orchestrator: Any | None = None
        self._unified_parser: Any | None = None
        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_error: str | None = None
        self.last_prompt: dict[str, Any] = {}
        self.last_raw_response: str | None = None

    async def decide(
        self,
        system_prompt: str,
        user_message: str,
        chart_b64: str | None = None,
    ) -> TradeDecision:
        """Send prompts to the configured AI provider and return a TradeDecision."""
        self.last_prompt = self._build_log_prompt(system_prompt, user_message, chart_b64)
        self.last_raw_response = None

        logger.debug(
            "LLM prompt — system:\n%s\n\nuser:\n%s",
            truncate_for_log(system_prompt),
            truncate_for_log(user_message),
        )

        messages = self._build_messages(system_prompt, user_message, chart_b64)

        if self.settings.provider == "azure":
            client = self._get_azure_client()
            result = await decide_azure(
                self.settings, client, messages,
                self.ALLOWED_SYMBOLS, self._primary_symbol,
                sleep_func=self._sleep_func,
            )
        else:
            orch = self._get_orchestrator()
            parser = self._get_unified_parser()
            result = await decide_multi_provider(
                self.settings, orch, parser, messages, chart_b64,
                self.ALLOWED_SYMBOLS, self._primary_symbol,
                sleep_func=self._sleep_func,
            )

        self._apply_result(result)

        logger.debug(
            "LLM decision — action=%s symbol=%s confidence=%.2f reasoning=%s",
            result.decision.action.value,
            result.decision.symbol,
            result.decision.confidence,
            result.decision.reasoning,
        )
        return result.decision

    async def send_prompt(
        self,
        prompt: str,
        system_message: str | None = None,
        prepared_messages: List[Dict[str, str]] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Send a text prompt and return the raw response string."""
        messages = prepared_messages or self._build_messages(
            system_message or "", prompt, None
        )
        if self.settings.provider == "azure":
            client = self._get_azure_client()
            response = await complete_azure(client, self.settings.azure_deployment, messages)
            return extract_content(response)

        orch = self._get_orchestrator()
        effective_provider = provider or self.settings.provider
        result = await orch.get_text_response(effective_provider, messages, model)
        return self._result_content(result)

    async def send_prompt_with_chart_analysis(
        self,
        prompt: str,
        chart_image: Union[io.BytesIO, bytes, str],
        system_message: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Send a prompt with a chart image for visual analysis."""
        if self.settings.provider == "azure":
            chart_b64 = self._to_b64(chart_image)
            messages = self._build_messages(system_message or "", prompt, chart_b64)
            client = self._get_azure_client()
            response = await complete_azure(client, self.settings.azure_deployment, messages)
            return extract_content(response)

        orch = self._get_orchestrator()
        msgs = self._build_messages(system_message or "", prompt, None)
        effective_provider = provider or self.settings.provider
        result = await orch.get_chart_response(effective_provider, msgs, chart_image, model)
        if not result.success:
            raise ValueError(f"Chart analysis failed: {result.error or 'invalid response'}")
        return self._result_content(result)

    async def close(self) -> None:
        """Close all open client sessions."""
        if self._orchestrator is not None:
            await close_orchestrator(self._orchestrator)

    def _apply_result(self, result: LLMCallResult) -> None:
        self.last_usage = result.usage
        self.last_raw_response = result.raw_response
        self.last_error = result.error

    def _get_azure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.settings)
        return self._client

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            self._orchestrator = build_orchestrator(self.settings)
        return self._orchestrator

    def _get_unified_parser(self) -> Any:
        if self._unified_parser is None:
            self._unified_parser = build_unified_parser()
        return self._unified_parser

    def _build_messages(
        self, system_prompt: str, user_message: str, chart_b64: str | None
    ) -> list[dict[str, Any]]:
        if chart_b64 and self.settings.model_supports_vision:
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
                        },
                    ],
                },
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _build_log_prompt(
        self, system_prompt: str, user_message: str, chart_b64: str | None
    ) -> dict[str, Any]:
        return {
            "system_prompt": truncate_for_log(system_prompt),
            "user_message": truncate_for_log(user_message),
            "chart_included": bool(chart_b64 and self.settings.model_supports_vision),
        }

    @staticmethod
    def _to_b64(chart_image: Union[io.BytesIO, bytes, str]) -> str:
        if isinstance(chart_image, str):
            return chart_image
        if isinstance(chart_image, io.BytesIO):
            return base64.b64encode(chart_image.getvalue()).decode()
        return base64.b64encode(chart_image).decode()

    @staticmethod
    def _result_content(result: Any) -> str:
        if hasattr(result, "content") and result.content:
            return result.content
        if hasattr(result, "response"):
            return str(result.response)
        return ""


__all__ = [
    "LLMManager",
    "ParseError",
    "ResponseShapeError",
    "_AnthropicResponseAdapter",
    "_normalize_azure_openai_endpoint",
    "_should_use_azure_foundry_anthropic_client",
]
