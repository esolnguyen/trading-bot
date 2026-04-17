"""Prompt-context helpers for VectorMemoryService.

Formats past experiences, synthetic insights, and anti-pattern warnings for
prompt injection.
"""

from __future__ import annotations

from typing import Any, Dict


def generate_synthetic_insight(meta: Dict[str, Any]) -> str:
    """Condense trade metadata into a one-line insight string."""
    parts = []

    context = meta.get("market_context", "")
    if context:
        parts.append(f"Entry: {context}")

    close_reason = meta.get("close_reason", "")
    if close_reason:
        parts.append(f"Exit: {close_reason}")

    sl_dist = meta.get("sl_distance_pct")
    tp_dist = meta.get("tp_distance_pct")
    if sl_dist is not None:
        parts.append(f"SL: {sl_dist:.2f}%")
    if tp_dist is not None:
        parts.append(f"TP: {tp_dist:.2f}%")

    rr = meta.get("rr_ratio")
    if rr is not None:
        parts.append(f"R/R: {rr:.1f}")

    max_profit = meta.get("max_profit_pct")
    max_dd = meta.get("max_drawdown_pct")
    if max_profit is not None and max_profit > 0:
        parts.append(f"MaxProfit: +{max_profit:.1f}%")
    if max_dd is not None and max_dd > 0:
        parts.append(f"MaxDD: -{max_dd:.1f}%")

    adx = meta.get("adx_at_entry")
    rsi = meta.get("rsi_at_entry")
    vol = meta.get("volatility_level")
    if adx is not None:
        parts.append(f"ADX: {adx:.0f}")
    if rsi is not None:
        parts.append(f"RSI: {rsi:.0f}")
    if vol:
        parts.append(f"Vol: {vol}")

    return " | ".join(parts) if parts else "No additional data"


def format_context_for_prompt(service: Any, current_context: str, k: int = 5) -> str:
    """Assemble the `RELEVANT PAST EXPERIENCES` block for prompts."""
    experiences = service.retrieve_similar_experiences(
        current_context, k, where={"outcome": {"$ne": "UPDATE"}}
    )
    if not experiences:
        return ""

    max_similarity = max(exp.similarity for exp in experiences)
    if len(experiences) <= 2 and max_similarity < 50:
        lines = [
            f"RELEVANT PAST EXPERIENCES (Context: {current_context}):",
            "",
            f"⚠️ LIMITED DATA: Only {len(experiences)} trade(s) with <50% similarity. "
            "Standard analysis recommended.",
            "",
        ]
    else:
        lines = [
            f"RELEVANT PAST EXPERIENCES (Context: {current_context}):",
            "",
        ]

    for i, exp in enumerate(experiences, 1):
        meta = exp.metadata
        outcome = meta.get("outcome", "UNKNOWN")
        pnl = meta.get("pnl_pct", 0)
        direction = meta.get("direction", "?")

        lines.append(f"{i}. [SIMILARITY {exp.similarity:.0f}%] {direction} trade")
        lines.append(f"   - Result: {outcome} ({pnl:+.2f}%)")
        lines.append(f"   - Context: {meta.get('market_context', 'N/A')}")
        reasoning = meta.get("reasoning", "")
        if reasoning and reasoning != "N/A":
            lines.append(f'   - Key Insight: "{reasoning}"')
        else:
            lines.append(
                f'   - Key Insight: "{generate_synthetic_insight(meta)}"'
            )
        lines.append("")

    anti_patterns = format_anti_patterns(service, k=2)
    if anti_patterns:
        lines.append("")
        lines.append(anti_patterns)

    return "\n".join(lines)


def compute_stats_for_context(
    service: Any, current_context: str, k: int = 20
) -> Dict[str, Any]:
    """Win-rate / avg-P&L summary for trades similar to `current_context`."""
    experiences = service.retrieve_similar_experiences(
        current_context, k, where={"outcome": {"$ne": "UPDATE"}}
    )
    if not experiences:
        return {"win_rate": 0, "avg_pnl": 0, "total_trades": 0}

    wins = sum(1 for e in experiences if e.metadata.get("outcome") == "WIN")
    pnls = [e.metadata.get("pnl_pct", 0) for e in experiences]

    return {
        "win_rate": (wins / len(experiences)) * 100 if experiences else 0,
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        "total_trades": len(experiences),
    }


def format_anti_patterns(service: Any, k: int = 3) -> str:
    """Format any `anti_pattern`-typed rules as an AVOID block."""
    rules = service.get_active_rules(n_results=k * 2)
    anti_rules = [
        r for r in rules if r.get("metadata", {}).get("rule_type") == "anti_pattern"
    ]
    if not anti_rules:
        return ""

    lines = ["⚠️ AVOID PATTERNS (learned from losses):"]
    for rule in anti_rules[:k]:
        lines.append(f"  - {rule['text']}")
    return "\n".join(lines)
