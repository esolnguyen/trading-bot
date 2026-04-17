"""Parsing, error types, and small utilities shared by the LLM manager paths."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from src.domain.trading import Action, TradeDecision

MAX_LOG_TEXT_CHARS = 4000


class ParseError(Exception):
    """Raised when model output cannot be mapped to TradeDecision."""


class ResponseShapeError(Exception):
    """Raised when the model response does not expose the expected content path."""


@dataclass
class LLMCallResult:
    """Return value of a single LLM call attempt loop.

    Carries the decision plus metadata that the manager mirrors back onto
    ``last_usage`` / ``last_error`` / ``last_raw_response`` for the trading loop.
    """

    decision: TradeDecision
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    raw_response: str | None = None
    error: str | None = None


def truncate_for_log(text: str | None, limit: int = MAX_LOG_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or (
        isinstance(status_code, int) and 500 <= status_code < 600
    )


def strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that some models add."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def fallback_decision(reason: str) -> TradeDecision:
    return TradeDecision(
        symbol="BTCUSDT",
        action=Action.HOLD,
        quantity=0.0,
        order_type="MARKET",
        price=None,
        reasoning=reason,
        confidence=0.0,
        timestamp=0,
        source="fallback_hold",
    )


def parse_decision_strict(
    content: str,
    allowed_symbols: Iterable[str],
    primary_symbol: str,
    source: str = "azure",
) -> TradeDecision:
    """Parse Azure's JSON response strictly, raising ParseError on schema issues."""
    allowed = frozenset(allowed_symbols)
    try:
        payload = json.loads(strip_markdown_fence(content))
    except json.JSONDecodeError as exc:
        raise ParseError(f"Malformed JSON (first 200 chars): {content[:200]!r}") from exc

    if "action" not in payload:
        raise ParseError("Missing required fields: ['action']")

    try:
        action = Action(payload["action"])
    except Exception as exc:
        raise ParseError("Invalid action") from exc

    if action == Action.HOLD:
        symbol = str(payload.get("symbol") or primary_symbol)
        if symbol not in allowed:
            symbol = primary_symbol
        return TradeDecision(
            symbol=symbol,
            action=Action.HOLD,
            quantity=0.0,
            order_type="MARKET",
            price=None,
            reasoning=str(payload.get("reasoning") or "model_hold_without_reason"),
            confidence=float(payload.get("confidence", 0.0)),
            timestamp=int(payload.get("timestamp", 0)),
            source=source,
        )

    required = {"symbol", "quantity", "order_type", "reasoning"}
    missing = required.difference(payload)
    if missing:
        raise ParseError(f"Missing required fields: {sorted(missing)}")

    symbol = str(payload["symbol"])
    if symbol not in allowed:
        raise ParseError("Invalid symbol")

    order_type = str(payload["order_type"])
    price = payload.get("price")
    if order_type == "LIMIT" and price is None:
        raise ParseError("LIMIT orders require price")

    return TradeDecision(
        symbol=symbol,
        action=action,
        quantity=float(payload["quantity"]),
        order_type=order_type,
        price=None if price is None else float(price),
        reasoning=str(payload["reasoning"]),
        confidence=float(payload.get("confidence", 0.0)),
        timestamp=int(payload.get("timestamp", 0)),
        source=source,
    )


def parse_decision_from_dict(
    payload: dict,
    allowed_symbols: Iterable[str],
    primary_symbol: str,
    source: str,
) -> TradeDecision:
    """Parse an already-extracted dict (from UnifiedParser) into TradeDecision.

    Unlike :func:`parse_decision_strict`, this never raises — it leans on
    the orchestrator's UnifiedParser upstream and returns a HOLD fallback
    when required fields are missing.
    """
    allowed = frozenset(allowed_symbols)
    if "action" not in payload:
        return fallback_decision("parse_error")

    try:
        action_str = str(payload["action"]).upper()
        if action_str in ("BUY", "SELL", "HOLD", "CLOSE", "UPDATE"):
            try:
                action = Action(action_str)
            except ValueError:
                action = Action.HOLD
        else:
            action = Action.HOLD
    except Exception:
        return fallback_decision("parse_error")

    if action == Action.HOLD:
        symbol = str(payload.get("symbol") or primary_symbol)
        if symbol not in allowed:
            symbol = primary_symbol
        return TradeDecision(
            symbol=symbol,
            action=Action.HOLD,
            quantity=0.0,
            order_type="MARKET",
            price=None,
            reasoning=str(payload.get("reasoning") or "hold"),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            timestamp=int(payload.get("timestamp", 0) or 0),
            source=source,
        )

    symbol = str(payload.get("symbol", primary_symbol))
    if symbol not in allowed:
        symbol = primary_symbol

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
        source=source,
    )
