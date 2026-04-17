"""Azure OpenAI / Azure AI Foundry (Anthropic) client setup and call handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from src.core.config import Settings
from src.domain.trading import TradeDecision
from src.infrastructure.ai.llm_parsing import (
    LLMCallResult,
    ParseError,
    ResponseShapeError,
    fallback_decision,
    is_retryable_error,
    parse_decision_strict,
    truncate_for_log,
)

logger = logging.getLogger(__name__)


class _AnthropicResponseAdapter:
    """Wrap an Anthropic Messages response to look like an OpenAI completion."""

    def __init__(self, response: Any) -> None:
        self._response = response
        content = ""
        if response.content:
            content = getattr(response.content[0], "text", "")
        self.choices = [
            type("_Choice", (), {"message": type("_Msg", (), {"content": content})()})()
        ]
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            self.usage = type("_Usage", (), {
                "prompt_tokens": getattr(usage_obj, "input_tokens", 0),
                "completion_tokens": getattr(usage_obj, "output_tokens", 0),
                "total_tokens": (
                    getattr(usage_obj, "input_tokens", 0)
                    + getattr(usage_obj, "output_tokens", 0)
                ),
            })()
        else:
            self.usage = None


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


def _should_use_azure_foundry_anthropic_client(endpoint: str, deployment: str) -> bool:
    normalized_endpoint = endpoint.strip().lower()
    normalized_deployment = deployment.strip().lower()

    if "services.ai.azure.com" not in normalized_endpoint:
        return False

    anthropic_markers = ("claude", "anthropic")
    return any(marker in normalized_endpoint for marker in anthropic_markers) or any(
        marker in normalized_deployment for marker in anthropic_markers
    )


def default_azure_client_factory(settings: Settings) -> Any:
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


async def complete_azure(
    client: Any, deployment: str, messages: list[dict[str, Any]]
) -> Any:
    """Run one Azure completion, normalising Anthropic responses to OpenAI shape."""
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        system_content = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        user_messages = [m for m in messages if m.get("role") != "system"]
        result = client.messages.create(
            model=deployment,
            max_tokens=512,
            system=system_content,
            messages=user_messages,
        )
        return _AnthropicResponseAdapter(result)

    complete = client.chat.completions.create
    request_kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 1,
    }
    try:
        result = complete(**request_kwargs, response_format={"type": "json_object"})
    except Exception as exc:
        if "Unsupported `response_format`" not in str(exc):
            raise
        result = complete(**request_kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def extract_content(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except Exception as exc:
        raise ResponseShapeError("Invalid response shape") from exc


def extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


async def decide_azure(
    settings: Settings,
    client: Any,
    messages: list[dict[str, Any]],
    allowed_symbols: frozenset[str],
    primary_symbol: str,
    *,
    sleep_func: Callable[[float], Awaitable[None] | Any] = asyncio.sleep,
) -> LLMCallResult:
    """Call Azure with exponential-backoff retry and parse a TradeDecision."""
    result = LLMCallResult(decision=fallback_decision("request_failed"))

    for attempt in range(3):
        try:
            response = await complete_azure(client, settings.azure_deployment, messages)
            result.usage = extract_usage(response)
            content = extract_content(response)
            result.raw_response = truncate_for_log(content)
            decision = parse_decision_strict(
                content, allowed_symbols, primary_symbol, source="azure"
            )
            result.decision = decision
            result.error = None
            return result
        except Exception as exc:  # noqa: BLE001
            if is_retryable_error(exc) and attempt < 2:
                result.error = str(exc)
                await sleep_func(2 * (2 ** attempt))
                continue
            result.error = str(exc)
            logger.warning(
                "LLM Azure fallback HOLD (attempt %s): %s | raw: %s",
                attempt + 1, exc, truncate_for_log(result.raw_response),
            )
            reason = "parse_error" if isinstance(exc, (ParseError, ResponseShapeError)) else "request_failed"
            result.decision = fallback_decision(reason)
            return result

    result.error = "request_failed"
    result.decision = fallback_decision("request_failed")
    return result


# Ignore unused-import warnings — retained for callers that previously imported
# these helpers directly from llm_manager.
__all__ = [
    "_AnthropicResponseAdapter",
    "_normalize_azure_openai_endpoint",
    "_should_use_azure_foundry_anthropic_client",
    "complete_azure",
    "decide_azure",
    "default_azure_client_factory",
    "extract_content",
    "extract_usage",
]

_ = TradeDecision  # keep import alive for type checkers
