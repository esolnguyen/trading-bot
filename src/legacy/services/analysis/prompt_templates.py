"""Prompt templates for the trading decision workflow."""

from __future__ import annotations


def build_system_prompt(
    max_order_usdt: float,
    regime_suffix: str = "",
    trading_symbols: list[str] | None = None,
    skills_block: str = "",
) -> str:
    """Return the system prompt contract for the trading LLM.

    Args:
        max_order_usdt: Maximum order size in USDT.
        regime_suffix: Optional macro regime instruction injected by CycleClassifier.
        trading_symbols: Allowed symbols list; defaults to BTCUSDT, ETHUSDT.
        skills_block: Pre-rendered markdown block of playbook skills, appended
            after the contract. Use ``load_skills(...)`` to build it.
    """
    symbols_str = ", ".join(trading_symbols) if trading_symbols else "BTCUSDT, ETHUSDT"
    base = (
        "You are an autonomous crypto trading agent.\n"
        f"Allowed symbols: {symbols_str} only.\n"
        f"Max order size: {max_order_usdt} USDT.\n"
        "Allowed order types: MARKET, LIMIT only.\n"
        "Output format: JSON only, no prose outside JSON.\n"
        "Choose the single best opportunity among the allowed symbols.\n"
        "Use HOLD only when both symbols lack a clear edge, the evidence is conflicting, or the setup is too weak.\n"
        "Do not use HOLD just because one symbol is neutral if the other has a clear setup.\n"
        'For BUY or SELL, include: {"action","symbol","quantity","order_type","reasoning"}.\n'
        'For HOLD, include: {"action":"HOLD","symbol","reasoning"}.\n'
        "Reasoning must briefly cite the strongest technical or contextual evidence."
    )
    prompt = base + regime_suffix
    if skills_block:
        prompt += "\n\n" + skills_block
    return prompt
