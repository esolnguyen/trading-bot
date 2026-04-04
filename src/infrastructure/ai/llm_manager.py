"""Multi-provider LLM manager with safe parsing and fallback behavior."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision

logger = logging.getLogger(__name__)


class LLMManager:
    """Send prompt context to the model and parse a safe trade decision.

    Supports Azure OpenAI (legacy path) and the full multi-provider chain
    via ProviderOrchestrator (Google, OpenRouter, LM Studio, BlockRun).
    The external ``decide()`` interface is unchanged for TradingLoop compatibility.
    """

    MAX_LOG_TEXT_CHARS = 4000

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
        # Azure client (used when provider == "azure")
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        # Multi-provider components (initialized lazily for non-Azure providers)
        self._orchestrator: Any | None = None
        self._unified_parser: Any | None = None
        # Public state read by TradingLoop
        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_error: str | None = None
        self.last_prompt: dict[str, Any] = {}
        self.last_raw_response: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            self._truncate_for_log(system_prompt),
            self._truncate_for_log(user_message),
        )

        if self.settings.provider == "azure":
            decision = await self._decide_azure(system_prompt, user_message, chart_b64)
        else:
            decision = await self._decide_multi_provider(system_prompt, user_message, chart_b64)

        logger.debug(
            "LLM decision — action=%s symbol=%s confidence=%.2f reasoning=%s",
            decision.action.value,
            decision.symbol,
            decision.confidence,
            decision.reasoning,
        )
        return decision

    async def send_prompt(
        self,
        prompt: str,
        system_message: str | None = None,
        prepared_messages: List[Dict[str, str]] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Send a text prompt and return the raw response string."""
        if self.settings.provider == "azure":
            messages = prepared_messages or self._build_messages(
                system_message or "", prompt, None
            )
            client = self._get_azure_client()
            response = await self._complete_azure(client, messages)
            return self._extract_content(response)

        orch = self._get_orchestrator()
        msgs = prepared_messages or self._build_messages(system_message or "", prompt, None)
        effective_provider = provider or self.settings.provider
        result = await orch.get_text_response(effective_provider, msgs, model)
        return await self._process_orchestrator_result(result)

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
            response = await self._complete_azure(client, messages)
            return self._extract_content(response)

        orch = self._get_orchestrator()
        msgs = self._build_messages(system_message or "", prompt, None)
        effective_provider = provider or self.settings.provider
        result = await orch.get_chart_response(effective_provider, msgs, chart_image, model)
        if not result.success:
            raise ValueError(f"Chart analysis failed: {result.error or 'invalid response'}")
        return await self._process_orchestrator_result(result)

    async def close(self) -> None:
        """Close all open client sessions."""
        try:
            if self._orchestrator is not None:
                clients = getattr(self._orchestrator, "_clients", None)
                if clients:
                    for attr in ("openrouter", "google", "google_paid", "lmstudio", "blockrun"):
                        client = getattr(clients, attr, None)
                        if client is not None and hasattr(client, "close"):
                            try:
                                await client.close()
                            except Exception:
                                pass
        except Exception as exc:
            logger.error("Error during LLMManager cleanup: %s", exc)

    # ------------------------------------------------------------------
    # Azure path
    # ------------------------------------------------------------------

    async def _decide_azure(
        self, system_prompt: str, user_message: str, chart_b64: str | None
    ) -> TradeDecision:
        client = self._get_azure_client()
        messages = self._build_messages(system_prompt, user_message, chart_b64)

        for attempt in range(3):
            try:
                response = await self._complete_azure(client, messages)
                self.last_usage = self._extract_usage(response)
                self.last_error = None
                content = self._extract_content(response)
                self.last_raw_response = self._truncate_for_log(content)
                return self._parse_decision(content)
            except Exception as exc:  # noqa: BLE001
                if self._is_retryable_error(exc) and attempt < 2:
                    self.last_error = str(exc)
                    await self._sleep_func(2 * (2 ** attempt))
                    continue
                self.last_error = str(exc)
                logger.warning(
                    "LLM Azure fallback HOLD (attempt %s): %s | raw: %s",
                    attempt + 1, exc, self._truncate_for_log(self.last_raw_response),
                )
                return self._fallback_decision("parse_error")

        self.last_error = "request_failed"
        return self._fallback_decision("request_failed")

    def _get_azure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.settings)
        return self._client

    async def _complete_azure(self, client: Any, messages: list[dict[str, Any]]) -> Any:
        # Anthropic SDK client (Azure AI Foundry path)
        if hasattr(client, "messages") and not hasattr(client, "chat"):
            system_content = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            user_messages = [m for m in messages if m.get("role") != "system"]
            result = client.messages.create(
                model=self.settings.azure_deployment,
                max_tokens=512,
                system=system_content,
                messages=user_messages,
            )
            # Wrap into an object that _extract_content and _extract_usage can read
            return _AnthropicResponseAdapter(result)

        # OpenAI SDK client (traditional Azure OpenAI path)
        complete = client.chat.completions.create
        request_kwargs: Dict[str, Any] = {
            "model": self.settings.azure_deployment,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 1,
        }
        try:
            result = complete(
                **request_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "Unsupported `response_format`" not in str(exc):
                raise
            result = complete(**request_kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # Multi-provider path
    # ------------------------------------------------------------------

    async def _decide_multi_provider(
        self, system_prompt: str, user_message: str, chart_b64: str | None
    ) -> TradeDecision:
        orch = self._get_orchestrator()
        parser = self._get_unified_parser()

        messages = self._build_messages(system_prompt, user_message, chart_b64)
        provider = self.settings.provider

        for attempt in range(3):
            try:
                if chart_b64 and self.settings.model_supports_vision:
                    result = await orch.get_chart_response(provider, messages, chart_b64)
                else:
                    result = await orch.get_text_response(provider, messages)

                if not result.success:
                    raise RuntimeError(result.error or "provider returned failure")

                content = _strip_markdown_fence(
                    result.content if hasattr(result, "content") else str(result.response)
                )
                self.last_raw_response = self._truncate_for_log(content)

                # Update usage from result
                usage = getattr(result, "usage", None)
                if usage:
                    self.last_usage = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }

                # Parse content via UnifiedParser → TradeDecision
                parsed = parser.parse_ai_response(content)
                self.last_error = None
                return self._parse_decision_from_dict(parsed)

            except Exception as exc:  # noqa: BLE001
                if self._is_retryable_error(exc) and attempt < 2:
                    self.last_error = str(exc)
                    await self._sleep_func(2 * (2 ** attempt))
                    continue
                self.last_error = str(exc)
                logger.warning(
                    "LLM multi-provider fallback HOLD (attempt %s): %s | raw: %s",
                    attempt + 1, exc, self._truncate_for_log(self.last_raw_response),
                )
                return self._fallback_decision("request_failed")

        return self._fallback_decision("request_failed")

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            from src.infrastructure.ai.provider_orchestrator import ProviderOrchestrator
            from src.infrastructure.ai.providers.google import GoogleAIClient
            from src.infrastructure.ai.providers.openrouter import OpenRouterClient
            from src.infrastructure.ai.providers.lmstudio import LMStudioClient
            from src.infrastructure.ai.providers.blockrun import BlockRunClient
            from src.infrastructure.ai.provider_types import ProviderClients

            s = self.settings
            _logger = logging.getLogger("LLMManager.providers")
            clients = ProviderClients(
                google=GoogleAIClient(settings=s, logger=_logger) if s.google_studio_api_key else None,
                google_paid=GoogleAIClient(settings=s, logger=_logger, use_paid=True) if s.google_studio_paid_api_key else None,
                openrouter=OpenRouterClient(settings=s, logger=_logger) if s.openrouter_api_key else None,
                lmstudio=LMStudioClient(settings=s, logger=_logger),
                blockrun=BlockRunClient(settings=s, logger=_logger) if s.blockrun_wallet_key else None,
            )
            self._orchestrator = ProviderOrchestrator(settings=s, logger=_logger, provider_clients=clients)
        return self._orchestrator

    def _get_unified_parser(self) -> Any:
        if self._unified_parser is None:
            from src.infrastructure.ai.unified_parser import UnifiedParser
            self._unified_parser = UnifiedParser(logger=logging.getLogger("UnifiedParser"))
        return self._unified_parser

    async def _process_orchestrator_result(self, result: Any) -> str:
        if hasattr(result, "content") and result.content:
            return result.content
        if hasattr(result, "response"):
            return str(result.response)
        return ""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_decision_from_dict(self, payload: dict) -> TradeDecision:
        """Parse a dict (already extracted from JSON) into TradeDecision."""
        if "action" not in payload:
            return self._fallback_decision("parse_error")

        try:
            action_str = str(payload["action"]).upper()
            # Map LLM_trader-style actions to the simple executor actions
            if action_str in ("BUY", "SELL", "HOLD", "CLOSE", "UPDATE"):
                try:
                    action = Action(action_str)
                except ValueError:
                    action = Action.HOLD
            else:
                action = Action.HOLD
        except Exception:
            return self._fallback_decision("parse_error")

        _primary = self._primary_symbol
        if action == Action.HOLD:
            symbol = str(payload.get("symbol") or _primary)
            if symbol not in self.ALLOWED_SYMBOLS:
                symbol = _primary
            return TradeDecision(
                symbol=symbol, action=Action.HOLD, quantity=0.0,
                order_type="MARKET", price=None,
                reasoning=str(payload.get("reasoning") or "hold"),
                confidence=float(payload.get("confidence", 0.0) or 0.0),
                timestamp=int(payload.get("timestamp", 0) or 0),
                source=self.settings.provider,
            )

        symbol = str(payload.get("symbol", _primary))
        if symbol not in self.ALLOWED_SYMBOLS:
            symbol = _primary

        order_type = str(payload.get("order_type", "MARKET"))
        price = payload.get("price")

        return TradeDecision(
            symbol=symbol,
            action=action,
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            order_type=order_type,
            price=None if price is None else float(price),
            reasoning=str(payload.get("reasoning", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            timestamp=int(payload.get("timestamp", 0) or 0),
            source=self.settings.provider,
        )

    def _parse_decision(self, content: str) -> TradeDecision:
        import json as _json
        try:
            payload = _json.loads(_strip_markdown_fence(content))
        except _json.JSONDecodeError as exc:
            raise ParseError(f"Malformed JSON (first 200 chars): {content[:200]!r}") from exc

        if "action" not in payload:
            raise ParseError("Missing required fields: ['action']")

        try:
            action = Action(payload["action"])
        except Exception as exc:
            raise ParseError("Invalid action") from exc

        _primary = self._primary_symbol
        if action == Action.HOLD:
            symbol = str(payload.get("symbol") or _primary)
            if symbol not in self.ALLOWED_SYMBOLS:
                symbol = _primary
            return TradeDecision(
                symbol=symbol, action=Action.HOLD, quantity=0.0,
                order_type="MARKET", price=None,
                reasoning=str(payload.get("reasoning") or "model_hold_without_reason"),
                confidence=float(payload.get("confidence", 0.0)),
                timestamp=int(payload.get("timestamp", 0)),
                source="azure",
            )

        required = {"symbol", "quantity", "order_type", "reasoning"}
        missing = required.difference(payload)
        if missing:
            raise ParseError(f"Missing required fields: {sorted(missing)}")

        symbol = str(payload["symbol"])
        if symbol not in self.ALLOWED_SYMBOLS:
            raise ParseError("Invalid symbol")

        order_type = str(payload["order_type"])
        price = payload.get("price")
        if order_type == "LIMIT" and price is None:
            raise ParseError("LIMIT orders require price")

        return TradeDecision(
            symbol=symbol, action=action,
            quantity=float(payload["quantity"]),
            order_type=order_type,
            price=None if price is None else float(price),
            reasoning=str(payload["reasoning"]),
            confidence=float(payload.get("confidence", 0.0)),
            timestamp=int(payload.get("timestamp", 0)),
            source="azure",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            "system_prompt": self._truncate_for_log(system_prompt),
            "user_message": self._truncate_for_log(user_message),
            "chart_included": bool(chart_b64 and self.settings.model_supports_vision),
        }

    @staticmethod
    def _to_b64(chart_image: Union[io.BytesIO, bytes, str]) -> str:
        import base64
        if isinstance(chart_image, str):
            return chart_image
        if isinstance(chart_image, io.BytesIO):
            return base64.b64encode(chart_image.getvalue()).decode()
        return base64.b64encode(chart_image).decode()

    def _extract_content(self, response: Any) -> str:
        try:
            return response.choices[0].message.content
        except Exception as exc:
            raise ResponseShapeError("Invalid response shape") from exc

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or (
            isinstance(status_code, int) and 500 <= status_code < 600
        )

    @classmethod
    def _truncate_for_log(cls, text: str | None) -> str | None:
        if text is None:
            return None
        if len(text) <= cls.MAX_LOG_TEXT_CHARS:
            return text
        return f"{text[:cls.MAX_LOG_TEXT_CHARS]}...[truncated]"

    @staticmethod
    def _fallback_decision(reason: str) -> TradeDecision:
        return TradeDecision(
            symbol="BTCUSDT", action=Action.HOLD, quantity=0.0,
            order_type="MARKET", price=None, reasoning=reason,
            confidence=0.0, timestamp=0, source="fallback_hold",
        )

    @staticmethod
    def _default_client_factory(settings: Settings) -> Any:
        endpoint = settings.azure_endpoint.strip().rstrip("/")

        # Azure AI Foundry exposes multiple API shapes. Claude deployments use
        # Anthropic Messages, while OpenAI-compatible models still work via the
        # Azure OpenAI client even on *.services.ai.azure.com endpoints.
        if _should_use_azure_foundry_anthropic_client(endpoint, settings.azure_deployment):
            try:
                import anthropic
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "anthropic is required for Claude deployments on Azure AI Foundry"
                ) from exc
            # Normalise to the base URL; SDK appends /v1/messages automatically
            for suffix in ("/anthropic/v1/messages", "/anthropic/v1", "/anthropic", "/models"):
                if endpoint.endswith(suffix):
                    endpoint = endpoint[: -len(suffix)]
                    break
            return anthropic.Anthropic(
                base_url=f"{endpoint}/anthropic",
                api_key=settings.azure_api_key,
                default_headers={"api-key": settings.azure_api_key},
            )

        # Traditional Azure OpenAI endpoint (*.openai.azure.com)
        try:
            from openai import AzureOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai is required to use LLMManager with Azure") from exc
        endpoint = _normalize_azure_openai_endpoint(endpoint)
        return AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=endpoint,
        )


class _AnthropicResponseAdapter:
    """Wrap an Anthropic Messages response to look like an OpenAI completion."""

    def __init__(self, response: Any) -> None:
        self._response = response
        content = ""
        if response.content:
            content = getattr(response.content[0], "text", "")
        # Mimic OpenAI response shape
        self.choices = [type("_Choice", (), {"message": type("_Msg", (), {"content": content})()})()]
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            self.usage = type("_Usage", (), {
                "prompt_tokens": getattr(usage_obj, "input_tokens", 0),
                "completion_tokens": getattr(usage_obj, "output_tokens", 0),
                "total_tokens": (getattr(usage_obj, "input_tokens", 0) + getattr(usage_obj, "output_tokens", 0)),
            })()
        else:
            self.usage = None


class ParseError(Exception):
    """Raised when model output cannot be mapped to TradeDecision."""


class ResponseShapeError(Exception):
    """Raised when the model response does not expose the expected content path."""


def _normalize_azure_openai_endpoint(endpoint: str) -> str:
    raw = endpoint.strip().rstrip("/")
    parts = urlsplit(raw)
    path = parts.path
    for suffix in ("/chat/completions", "/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    if not path:
        path = ""
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that some models add."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence line (```json or ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _should_use_azure_foundry_anthropic_client(endpoint: str, deployment: str) -> bool:
    normalized_endpoint = endpoint.strip().lower()
    normalized_deployment = deployment.strip().lower()

    if "services.ai.azure.com" not in normalized_endpoint:
        return False

    anthropic_markers = ("claude", "anthropic")
    return any(marker in normalized_endpoint for marker in anthropic_markers) or any(
        marker in normalized_deployment for marker in anthropic_markers
    )
