"""LangGraph ReAct agent + the decision schema it produces.

``TradingDecision`` is both the return type the runtime cares about
and the ``response_format`` the agent is constrained to emit. When
LangGraph finishes its tool-calling loop, the structured answer is in
``result['structured_response']``.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from ..config import TradingBotSettings
from .prompts import SYSTEM_PROMPT

Side = Literal["LONG", "SHORT", "HOLD"]


class TradingDecision(BaseModel):
    """Final structured decision emitted per tick."""

    symbol: str
    side: Side
    conviction: int = Field(ge=1, le=10)
    rationale: str = Field(
        description="One paragraph — which signals agreed, which conflicted, "
                    "and what would invalidate the thesis."
    )
    stop_loss_pct: float | None = Field(
        default=None,
        description="Percent from entry, positive number (e.g. 1.5 for 1.5%). "
                    "Required when side is LONG or SHORT; null for HOLD.",
    )
    take_profit_pct: float | None = Field(
        default=None,
        description="Percent from entry, positive number. Required when side "
                    "is LONG or SHORT; null for HOLD.",
    )


def build_agent(settings: TradingBotSettings, tools: list[BaseTool]) -> Any:
    """Compile the ReAct graph bound to the given tool list."""
    model = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        response_format=TradingDecision,
    )
