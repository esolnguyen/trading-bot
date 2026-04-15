"""LLM decision engine — thin re-export shim.

Logic lives in src/services/llm_trader/llm_decision.py.
The Action enum is shared from src/domain/trading/models.py.
"""
from src.domain.trading.models import Action as Action
from src.services.llm_trader.llm_decision import (
    LLMDecision as LLMDecision,
    LLMDecisionEngine as LLMDecisionEngine,
)

__all__ = ["Action", "LLMDecision", "LLMDecisionEngine"]
