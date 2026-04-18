"""LLM manager package: Azure + multi-provider orchestration behind one facade."""

from .manager import (
    LLMManager,
    _normalize_azure_openai_endpoint,
    _should_use_azure_foundry_anthropic_client,
)

__all__ = [
    "LLMManager",
    "_normalize_azure_openai_endpoint",
    "_should_use_azure_foundry_anthropic_client",
]
