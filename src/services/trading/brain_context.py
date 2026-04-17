"""Brain context-string builders — shared by storage and prompt retrieval."""

from __future__ import annotations

from typing import Any, Callable


def build_rich_context_string(
    trend_direction: str = "NEUTRAL",
    adx: float = 0,
    volatility_level: str = "MEDIUM",
    rsi_level: str = "NEUTRAL",
    macd_signal: str = "NEUTRAL",
    volume_state: str = "NORMAL",
    bb_position: str = "MIDDLE",
    is_weekend: bool = False,
    market_sentiment: str = "NEUTRAL",
    order_book_bias: str = "BALANCED",
) -> str:
    """Build the semantic context string used for both vector storage and query.

    Keeping the exact same shape in both places maximises cosine similarity
    between "store" and "retrieve" paths.
    """
    adx_label = "High ADX" if adx >= 25 else "Low ADX" if adx < 20 else "Medium ADX"

    parts = [trend_direction, adx_label, f"{volatility_level} Volatility"]

    if rsi_level != "NEUTRAL":
        parts.append(f"RSI {rsi_level}")
    if macd_signal != "NEUTRAL":
        parts.append(f"MACD {macd_signal}")
    if volume_state != "NORMAL":
        parts.append(f"Volume {volume_state}")
    if bb_position != "MIDDLE":
        parts.append(f"Price at BB {bb_position}")
    if is_weekend:
        parts.append("Weekend Low Volume")
    if market_sentiment not in ("NEUTRAL", ""):
        parts.append(f"Sentiment {market_sentiment}")
    if order_book_bias not in ("BALANCED", ""):
        parts.append(f"OrderBook {order_book_bias}")

    return " + ".join(parts)


def get_vector_context(
    vector_memory: Any,
    *,
    trend_direction: str = "NEUTRAL",
    adx: float = 0,
    volatility_level: str = "MEDIUM",
    rsi_level: str = "NEUTRAL",
    macd_signal: str = "NEUTRAL",
    volume_state: str = "NORMAL",
    bb_position: str = "MIDDLE",
    is_weekend: bool = False,
    market_sentiment: str = "NEUTRAL",
    order_book_bias: str = "BALANCED",
    k: int = 5,
) -> str:
    """Semantic-search past trades for the given market context and format them."""
    context_query = build_rich_context_string(
        trend_direction=trend_direction,
        adx=adx,
        volatility_level=volatility_level,
        rsi_level=rsi_level,
        macd_signal=macd_signal,
        volume_state=volume_state,
        bb_position=bb_position,
        is_weekend=is_weekend,
        market_sentiment=market_sentiment,
        order_book_bias=order_book_bias,
    )

    vector_context = vector_memory.get_context_for_prompt(context_query, k)
    if not vector_context:
        return ""

    stats = vector_memory.get_stats_for_context(context_query, k=20)
    if stats["total_trades"] > 0:
        vector_context += (
            f"### Learned Stats for This Context:\n"
            f"- Win Rate in similar conditions: {stats['win_rate']:.0f}% "
            f"({stats['total_trades']} trades)\n"
            f"- Avg P&L: {stats['avg_pnl']:+.2f}%\n"
        )
    return vector_context


def build_brain_prompt_context(
    vector_memory: Any,
    cached_stats: Callable[[str, Callable[[], Any]], Any],
    *,
    trend_direction: str = "NEUTRAL",
    adx: float = 0,
    volatility_level: str = "MEDIUM",
    rsi_level: str = "NEUTRAL",
    macd_signal: str = "NEUTRAL",
    volume_state: str = "NORMAL",
    bb_position: str = "MIDDLE",
    is_weekend: bool = False,
    market_sentiment: str = "NEUTRAL",
    order_book_bias: str = "BALANCED",
) -> str:
    """Full brain context for prompt injection: calibration + vector + rules."""
    lines: list[str] = []

    exp_count = vector_memory.trade_count  # Excludes UPDATE entries
    if exp_count > 0:
        lines.extend([
            "",
            f"## Trading Brain ({exp_count} closed trades)",
            "",
            "### Confidence Calibration:",
        ])

        conf_stats = cached_stats("confidence", vector_memory.compute_confidence_stats)
        for level in ["HIGH", "MEDIUM", "LOW"]:
            stats = conf_stats.get(level, {})
            if stats.get("total_trades", 0) > 0:
                lines.append(
                    f"- {level} Confidence: Win Rate {stats['win_rate']:.0f}% "
                    f"({stats['winning_trades']}/{stats['total_trades']} trades) | "
                    f"Avg P&L: {stats['avg_pnl_pct']:+.2f}%"
                )

        recommendation = vector_memory.get_confidence_recommendation()
        if recommendation:
            lines.append(f"  → INSIGHT: {recommendation}")

        direction_bias = vector_memory.get_direction_bias()
        if direction_bias:
            lines.extend([
                "",
                "### Direction Bias Check:",
                f"- Historical trades: {direction_bias['long_count']} LONG, "
                f"{direction_bias['short_count']} SHORT",
            ])
            if direction_bias["short_count"] == 0:
                lines.append(
                    "- ⚠️ NO SHORT TRADES IN HISTORY: Consider SHORT opportunities "
                    "more carefully; lack of data means you may be missing valid setups."
                )

    vector_context = get_vector_context(
        vector_memory,
        trend_direction=trend_direction, adx=adx, volatility_level=volatility_level,
        rsi_level=rsi_level, macd_signal=macd_signal, volume_state=volume_state,
        bb_position=bb_position, is_weekend=is_weekend,
        market_sentiment=market_sentiment, order_book_bias=order_book_bias, k=5,
    )
    if vector_context:
        lines.extend(["", vector_context])

    has_limited_data = "⚠️ LIMITED DATA" in vector_context if vector_context else False

    if lines and not has_limited_data:
        lines.extend([
            "",
            "### Apply Insights (CoT Step 6 - Historical Evidence):",
            "- MANDATORY: If win rate in similar conditions <50%, reduce your confidence "
            "by 10 points and state this adjustment.",
            '- MANDATORY: If AVOID PATTERNS match current conditions (>50% similarity), '
            'state "⚠️ ANTI-PATTERN MATCH" and justify any override.',
            "- Weight recent wins higher. Check for pattern repetition that led to losses.",
            "",
        ])
    elif lines and has_limited_data:
        lines.extend([
            "",
            "NOTE: Limited historical data available. Rely on standard technical analysis "
            "for this decision.",
            "",
        ])

    adx_label = "High ADX" if adx >= 25 else "Low ADX" if adx < 20 else "Medium ADX"
    rule_context = f"{trend_direction} + {adx_label} + {volatility_level} Volatility"

    semantic_rules = vector_memory.get_relevant_rules(
        current_context=rule_context, n_results=3
    )
    if semantic_rules:
        lines.append("### Learned Trading Rules (relevant to current conditions):")
        for rule in semantic_rules:
            similarity = rule.get("similarity", 0)
            lines.append(f"- [{similarity:.0f}% match] {rule['text']}")
        lines.append("")

    return "\n".join(lines)
