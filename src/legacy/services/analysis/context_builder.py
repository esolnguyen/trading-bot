"""Assemble system and user prompts for the decision model."""

from __future__ import annotations

from typing import Any, Optional

from src.mcp_servers.config import Settings
from src.mcp_servers.shared.domain.analysis import PatternResult, TechnicalAnalysis
from src.mcp_servers.shared.domain.market import MarketSnapshot
from src.legacy.services.analysis.prompt_templates import build_system_prompt
from src.legacy.services.llm_trader.skills_loader import load_skills


class ContextBuilder:
    """Build the model-ready prompt pair from current market context."""

    MAX_USER_MESSAGE_CHARS = 8000

    def __init__(
        self,
        settings: Settings,
        *,
        brain_service: Optional[Any] = None,
        trading_strategy: Optional[Any] = None,
        memory_service: Optional[Any] = None,
        cycle_classifier: Optional[Any] = None,
        minimal_context: bool = False,
    ) -> None:
        self.settings = settings
        self._brain_service = brain_service
        self._trading_strategy = trading_strategy
        self._memory_service = memory_service
        self._cycle_classifier = cycle_classifier
        self._regime_suffix: str = ""  # updated by TradingLoop daily
        # When True, strip indicator/ML/RAG/memory sections from the user
        # message — the LLM gets chart + price + skills only. Used by the
        # llm_skills engine for a pure price-action decision.
        self._minimal_context = minimal_context
        self._skills_block = load_skills(list(settings.trader_skills or []))

    def set_regime_suffix(self, suffix: str) -> None:
        """Public setter for regime suffix, called by TradingLoop on daily refresh."""
        self._regime_suffix = suffix

    def build(
        self,
        snapshots: dict[str, MarketSnapshot],
        analyses: dict[str, TechnicalAnalysis],
        patterns: dict[str, PatternResult],
        rag_context: str,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> tuple[str, str]:
        system_prompt = build_system_prompt(
            self.settings.max_order_usdt,
            self._regime_suffix,
            trading_symbols=list(getattr(self.settings, "trading_symbols", None) or []),
            skills_block=self._skills_block,
        )
        user_message = self._build_user_message(
            snapshots,
            analyses,
            patterns,
            rag_context,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
            ml_context=ml_context,
        )
        return system_prompt, user_message

    def _build_user_message(
        self,
        snapshots: dict[str, MarketSnapshot],
        analyses: dict[str, TechnicalAnalysis],
        patterns: dict[str, PatternResult],
        rag_context: str,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> str:
        symbols = list(snapshots.keys())

        # Build each section as (priority, label, content) — lower number = higher priority.
        # Sections that fit within MAX_USER_MESSAGE_CHARS are included in full;
        # those that don't are replaced with an [OMITTED] placeholder.
        sections: list[tuple[int, str, str]] = []

        # Minimal mode (llm_skills engine): price tape + chart image only.
        # Indicators, patterns, ML, brain, RAG, and memory are stripped so the
        # LLM makes a pure price-action decision guided by the skills block.
        if self._minimal_context:
            price_lines = ["## Market Snapshot"] + [
                self._price_only_line(s, snapshots) for s in symbols
            ]
            sections.append((0, "Market Snapshot", "\n".join(price_lines)))

            if position_context:
                sections.append(
                    (
                        2,
                        "Current Position",
                        f"## Current Position\n{position_context.strip()}",
                    )
                )

            if self.settings.model_supports_vision:
                chart_parts = [self._vision_section(s, patterns) for s in symbols]
                chart_block = "".join(chart_parts).rstrip()
                if chart_block:
                    sections.append(
                        (7, "Chart Images", f"## Chart Images\n{chart_block}")
                    )
        else:
            market_lines = ["## Market Snapshot"] + [
                self._market_line(s, snapshots, analyses) for s in symbols
            ]
            sections.append((0, "Market Snapshot", "\n".join(market_lines)))

            pattern_lines = ["## Patterns Detected"] + [
                self._pattern_line(s, patterns) for s in symbols
            ]
            sections.append((1, "Patterns Detected", "\n".join(pattern_lines)))

            if position_context:
                sections.append(
                    (
                        2,
                        "Current Position",
                        f"## Current Position\n{position_context.strip()}",
                    )
                )

            if brain_context:
                sections.append(
                    (
                        3,
                        "Adaptive Brain Insights",
                        f"## Adaptive Brain Insights\n{brain_context.strip()}",
                    )
                )

            if ml_context:
                sections.append((3, "ML Context", ml_context.strip()))

            if dynamic_thresholds:
                threshold_lines = ["## Dynamic Thresholds"] + [
                    f"- {k}: {v}" for k, v in dynamic_thresholds.items()
                ]
                sections.append((4, "Dynamic Thresholds", "\n".join(threshold_lines)))

            if memory_context:
                sections.append(
                    (
                        5,
                        "Trading History",
                        f"## Trading History\n{memory_context.strip()}",
                    )
                )

            sections.append((6, "RAG Context", f"## RAG Context\n{rag_context}"))

            if self.settings.model_supports_vision:
                chart_parts = [self._vision_section(s, patterns) for s in symbols]
                chart_block = "".join(chart_parts).rstrip()
                if chart_block:
                    sections.append(
                        (7, "Chart Images", f"## Chart Images\n{chart_block}")
                    )

        footer = "## Your Decision\nRespond with a single JSON object."

        # Fill sections from highest priority downward, respecting char budget.
        budget = self.MAX_USER_MESSAGE_CHARS - len(footer) - 4  # 4 for "\n\n"
        included: list[str] = []
        for _, label, content in sorted(sections, key=lambda x: x[0]):
            if len(content) <= budget:
                included.append(content)
                budget -= len(content) + 2  # "\n\n" separator
            else:
                included.append(
                    f"## {label}\n[OMITTED: section too large to fit in context window]"
                )
                budget -= len(included[-1]) + 2

        return "\n\n".join(included) + "\n\n" + footer

    @staticmethod
    def _price_only_line(
        symbol: str,
        snapshots: dict[str, MarketSnapshot],
    ) -> str:
        """Minimal-context version of _market_line — price + funding/OI only.

        Drops the TechnicalAnalysis signal so the LLM forms its own read from
        the chart + skills rather than anchoring on a precomputed indicator.
        """
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            return f"{symbol}: data unavailable"
        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        parts = [
            f"{display_symbol}: ${snapshot.price:.2f} ({snapshot.change_24h_pct:.2f}% 24h)"
        ]
        if snapshot.funding_rate is not None:
            fr_pct = snapshot.funding_rate * 100
            bias = "longs paying" if fr_pct > 0 else "shorts paying"
            parts.append(f"FR: {fr_pct:+.4f}%/8h ({bias})")
        if snapshot.open_interest is not None:
            oi_b = snapshot.open_interest / 1_000_000_000
            parts.append(f"OI: ${oi_b:.2f}B")
        return " | ".join(parts)

    @staticmethod
    def _market_line(
        symbol: str,
        snapshots: dict[str, MarketSnapshot],
        analyses: dict[str, TechnicalAnalysis],
    ) -> str:
        snapshot = snapshots.get(symbol)
        analysis = analyses.get(symbol)
        if snapshot is None or analysis is None:
            return f"{symbol}: data unavailable"
        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        parts = [
            f"{display_symbol}: ${snapshot.price:.2f} ({snapshot.change_24h_pct:.2f}% 24h)",
            f"Signal: {analysis.signal.value}",
        ]
        if snapshot.funding_rate is not None:
            fr_pct = snapshot.funding_rate * 100
            bias = "longs paying" if fr_pct > 0 else "shorts paying"
            parts.append(f"FR: {fr_pct:+.4f}%/8h ({bias})")
        if snapshot.open_interest is not None:
            oi_b = snapshot.open_interest / 1_000_000_000
            parts.append(f"OI: ${oi_b:.2f}B")
        parts.append(analysis.reasoning)
        return " | ".join(parts)

    @staticmethod
    def _pattern_line(symbol: str, patterns: dict[str, PatternResult]) -> str:
        pattern_result = patterns.get(symbol)
        if pattern_result is None:
            return f"{symbol}: no pattern data"
        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        pattern_text = (
            ", ".join(pattern_result.patterns) if pattern_result.patterns else "none"
        )
        return (
            f"{display_symbol}: {pattern_text} | "
            f"Support: {pattern_result.support} | Resistance: {pattern_result.resistance}"
        )

    @staticmethod
    def _vision_section(symbol: str, patterns: dict[str, PatternResult]) -> str:
        pattern_result = patterns.get(symbol)
        if pattern_result is None or not pattern_result.chart_png_b64:
            return ""
        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        return f"{display_symbol} Chart (base64 PNG): {pattern_result.chart_png_b64}\n"
