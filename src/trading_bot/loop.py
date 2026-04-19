"""One-shot decision cycle.

A tick = one pass of (spawn MCP clients → pull tools → run agent →
emit a ``TradingDecision`` per symbol → tear down). The loop never
owns state beyond the current tick; persistence (open positions,
cooldowns) will live in a future ``state.py`` once Phase 7 lands.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from .agent import TradingDecision, build_agent, build_mcp_tools
from .config import TradingBotSettings

logger = logging.getLogger(__name__)


def _tick_prompt(symbol: str, timeframe: str) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        f"It is {now}. Decide LONG/SHORT/HOLD for {symbol} on the "
        f"{timeframe} timeframe. Use the MCP tools to gather evidence, "
        f"cross-check at least two signals, then submit a TradingDecision."
    )


async def run_cycle(
    settings: TradingBotSettings,
) -> list[TradingDecision]:
    """Run one decision cycle across every symbol; return the decisions."""
    client, tools = await build_mcp_tools()
    decisions: list[TradingDecision] = []
    try:
        agent = build_agent(settings, tools)
        for symbol in settings.trading_symbols:
            prompt = _tick_prompt(symbol, settings.primary_timeframe)
            logger.info("tick: %s", symbol)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": settings.llm_max_iterations * 2},
            )
            decision = result.get("structured_response")
            if not isinstance(decision, TradingDecision):
                logger.warning("agent returned no structured_response for %s", symbol)
                continue
            decisions.append(decision)
            _log_decision(settings, decision)
    finally:
        await _close_client(client)
    return decisions


def _log_decision(settings: TradingBotSettings, decision: TradingDecision) -> None:
    gated = decision.conviction < settings.min_conviction and decision.side != "HOLD"
    prefix = "[would-trade]" if settings.bot_mode == "dry_run" else "[decision]"
    logger.info(
        "%s %s %s conviction=%d sl=%s tp=%s%s — %s",
        prefix,
        decision.symbol,
        decision.side,
        decision.conviction,
        decision.stop_loss_pct,
        decision.take_profit_pct,
        "  (gated: below min_conviction)" if gated else "",
        decision.rationale,
    )


async def _close_client(client: object) -> None:
    """Best-effort shutdown of the MCP multi-server client.

    The adapter API has churned — some versions expose ``aclose``,
    others rely on async-context-manager semantics or the GC.
    """
    for attr in ("aclose", "close"):
        fn = getattr(client, attr, None)
        if fn is None:
            continue
        try:
            result = fn()
            if hasattr(result, "__await__"):
                await result
            return
        except Exception:  # noqa: BLE001
            logger.debug("client %s() raised during shutdown", attr, exc_info=True)
            return
