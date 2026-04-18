"""Standalone LLM-driven futures trader.

Self-contained trading path: fetches candles, asks an LLM for a BUY/SELL/HOLD
decision, and places Binance futures orders with SL/TP brackets. Independent
of the main trading loop in src.services.trading.
"""

from .futures_trader import FuturesTrader
from .llm_decision import LLMDecision, LLMDecisionEngine
from .safety_tracker import SafetyTracker
from .skills_loader import available_skills, load_skills

__all__ = [
    "FuturesTrader",
    "LLMDecision",
    "LLMDecisionEngine",
    "SafetyTracker",
    "available_skills",
    "load_skills",
]
