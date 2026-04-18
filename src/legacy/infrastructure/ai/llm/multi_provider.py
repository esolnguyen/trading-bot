"""Multi-provider LLM dispatch (Google, OpenRouter, LM Studio, BlockRun)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from src.mcp_servers.config import Settings
from .parsing import (
    LLMCallResult,
    fallback_decision,
    is_retryable_error,
    parse_decision_from_dict,
    strip_markdown_fence,
    truncate_for_log,
)

logger = logging.getLogger(__name__)


def build_orchestrator(settings: Settings) -> Any:
    """Construct a ProviderOrchestrator with every configured provider."""
    from src.mcp_servers.shared.infrastructure.ai.provider_orchestrator import (
        ProviderOrchestrator,
    )
    from src.mcp_servers.shared.infrastructure.ai.providers.google import GoogleAIClient
    from src.mcp_servers.shared.infrastructure.ai.providers.openrouter import (
        OpenRouterClient,
    )
    from src.mcp_servers.shared.infrastructure.ai.providers.lmstudio import (
        LMStudioClient,
    )
    from src.mcp_servers.shared.infrastructure.ai.providers.blockrun import (
        BlockRunClient,
    )
    from src.mcp_servers.shared.infrastructure.ai.provider_types import ProviderClients

    _logger = logging.getLogger("LLMManager.providers")
    clients = ProviderClients(
        google=(
            GoogleAIClient(settings=settings, logger=_logger)
            if settings.google_studio_api_key
            else None
        ),
        google_paid=(
            GoogleAIClient(settings=settings, logger=_logger, use_paid=True)
            if settings.google_studio_paid_api_key
            else None
        ),
        openrouter=(
            OpenRouterClient(settings=settings, logger=_logger)
            if settings.openrouter_api_key
            else None
        ),
        lmstudio=LMStudioClient(settings=settings, logger=_logger),
        blockrun=(
            BlockRunClient(settings=settings, logger=_logger)
            if settings.blockrun_wallet_key
            else None
        ),
    )
    return ProviderOrchestrator(
        settings=settings, logger=_logger, provider_clients=clients
    )


def build_unified_parser() -> Any:
    from src.mcp_servers.shared.infrastructure.ai.unified_parser import UnifiedParser

    return UnifiedParser(logger=logging.getLogger("UnifiedParser"))


async def decide_multi_provider(
    settings: Settings,
    orchestrator: Any,
    parser: Any,
    messages: list[dict[str, Any]],
    chart_b64: str | None,
    allowed_symbols: frozenset[str],
    primary_symbol: str,
    *,
    sleep_func: Callable[[float], Awaitable[None] | Any] = asyncio.sleep,
) -> LLMCallResult:
    """Send prompts through the multi-provider orchestrator and parse a decision."""
    result = LLMCallResult(decision=fallback_decision("request_failed"))
    provider = settings.provider

    for attempt in range(3):
        try:
            if chart_b64 and settings.model_supports_vision:
                provider_result = await orchestrator.get_chart_response(
                    provider, messages, chart_b64
                )
            else:
                provider_result = await orchestrator.get_text_response(
                    provider, messages
                )

            if not provider_result.success:
                raise RuntimeError(provider_result.error or "provider returned failure")

            content = strip_markdown_fence(
                provider_result.content
                if hasattr(provider_result, "content")
                else str(provider_result.response)
            )
            result.raw_response = truncate_for_log(content)

            usage = getattr(provider_result, "usage", None)
            if usage:
                result.usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }

            parsed = parser.parse_ai_response(content)
            result.decision = parse_decision_from_dict(
                parsed, allowed_symbols, primary_symbol, source=settings.provider
            )
            result.error = None
            return result

        except Exception as exc:  # noqa: BLE001
            if is_retryable_error(exc) and attempt < 2:
                result.error = str(exc)
                await sleep_func(2 * (2**attempt))
                continue
            result.error = str(exc)
            logger.warning(
                "LLM multi-provider fallback HOLD (attempt %s): %s | raw: %s",
                attempt + 1,
                exc,
                truncate_for_log(result.raw_response),
            )
            result.decision = fallback_decision("request_failed")
            return result

    result.decision = fallback_decision("request_failed")
    return result


async def close_orchestrator(orchestrator: Any) -> None:
    """Close every open provider client attached to an orchestrator."""
    try:
        clients = getattr(orchestrator, "_clients", None)
        if not clients:
            return
        for attr in ("openrouter", "google", "google_paid", "lmstudio", "blockrun"):
            client = getattr(clients, attr, None)
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.error("Error during orchestrator cleanup: %s", exc)
