"""Decision mixin for TradingLoop: signal_scorer / LLM / HTF confirmation gate."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.domain.analysis import Signal
from src.domain.trading import Action, TradeDecision


class DecisionsMixin:
    """Provides `_decide_per_symbol`, `_decide_all_symbols_single_call`, `_decide`."""

    async def _decide_per_symbol(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        patterns: dict[str, Any],
        rag_per_symbol: dict[str, str],
        htf_analyses: dict[str, Any] | None,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> dict[str, TradeDecision]:
        """Run LLM decisions for all symbols.

        Fix #8: when ``single_symbol_decision=True`` (SINGLE_SYMBOL_DECISION=true),
        all symbols are evaluated in a *single* LLM call — the model sees cross-asset
        context and picks the best opportunity, halving API cost for 2-symbol setups.
        When False (default), one call per symbol runs in parallel (legacy behaviour).
        """
        # Signal scorer path — deterministic, no LLM call
        if self.signal_scorer is not None:
            decisions: dict[str, TradeDecision] = {}
            for symbol in snapshots:
                # Determine current position direction for this symbol
                pos_dir: str | None = None
                if self.trading_strategy and getattr(self.trading_strategy, "symbol", None) == symbol:
                    strat_pos = getattr(self.trading_strategy, "current_position", None)
                    if strat_pos:
                        pos_dir = strat_pos.direction
                # Fallback: always check _open_positions if strategy didn't provide direction
                if pos_dir is None and symbol in self._open_positions:
                    raw_dir = self._open_positions[symbol].get("direction", "")
                    pos_dir = "LONG" if raw_dir == "BUY" else ("SHORT" if raw_dir == "SELL" else None)

                htf_sig = None
                if htf_analyses:
                    htf_a = htf_analyses.get(symbol)
                    if htf_a is not None:
                        htf_sig = htf_a.signal

                decision = self.signal_scorer.decide(
                    symbol,
                    snapshots[symbol],
                    analyses[symbol],
                    regime=self._current_regime,
                    htf_signal=htf_sig,
                    position_direction=pos_dir,
                )

                # Re-entry cooldown: suppress new entries if the symbol
                # was closed too recently (prevents rapid-fire flipping).
                cooldown = getattr(self.settings, "reentry_cooldown_cycles", 3)
                if (
                    cooldown > 0
                    and decision.action in (Action.BUY, Action.SELL)
                    and pos_dir is None  # only applies to new entries
                    and symbol in self._last_close_cycle
                    and (self._cycle - self._last_close_cycle[symbol]) < cooldown
                ):
                    remaining = cooldown - (self._cycle - self._last_close_cycle[symbol])
                    self.logger.info(
                        "[cooldown] %s suppressed %s — %d cycle(s) remaining before re-entry",
                        symbol, decision.action.value, remaining,
                    )
                    decision = TradeDecision(
                        symbol=symbol, action=Action.HOLD,
                        confidence=0.0,
                        reasoning=f"cooldown ({remaining} cycles left) | {decision.reasoning}",
                        source="signal_scorer",
                    )

                decisions[symbol] = decision
            return decisions

        # LLM paths below
        if getattr(self.settings, "single_symbol_decision", False) and len(snapshots) > 1:
            return await self._decide_all_symbols_single_call(
                snapshots, analyses, patterns, rag_per_symbol, htf_analyses,
                position_context=position_context,
                memory_context=memory_context,
                brain_context=brain_context,
                dynamic_thresholds=dynamic_thresholds,
                ml_context=ml_context,
            )

        async def _decide_one(symbol: str) -> TradeDecision:
            snap = {symbol: snapshots[symbol]}
            ana = {symbol: analyses[symbol]}
            pat = {symbol: patterns[symbol]}
            rag = rag_per_symbol.get(symbol, "=== NO CONTEXT AVAILABLE ===")
            # Per-symbol HTF thresholds (filter to this symbol only)
            sym_htf = None
            if htf_analyses:
                htf_val = htf_analyses.get(symbol)
                sym_htf = {symbol: htf_val} if htf_val is not None else {}
            system_prompt, user_message = self.builder.build(
                snap, ana, pat, rag,
                position_context=position_context,
                memory_context=memory_context,
                brain_context=brain_context,
                dynamic_thresholds=dynamic_thresholds,
                ml_context=ml_context,
            )
            decision = await self._decide(system_prompt, user_message, sym_htf)
            # Enforce correct symbol — LLM may omit it and the parser defaults to the primary symbol.
            if decision.symbol != symbol:
                decision = TradeDecision(
                    symbol=symbol,
                    action=decision.action,
                    quantity=decision.quantity,
                    order_type=decision.order_type,
                    price=decision.price,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence,
                    timestamp=decision.timestamp,
                    source=decision.source,
                )
            return decision

        results = await asyncio.gather(*[_decide_one(sym) for sym in snapshots])
        return dict(zip(snapshots.keys(), results))

    async def _decide_all_symbols_single_call(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        patterns: dict[str, Any],
        rag_per_symbol: dict[str, str],
        htf_analyses: dict[str, Any] | None,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> dict[str, TradeDecision]:
        """Fix #8: single LLM call for all symbols — pick the best opportunity.

        Per-symbol RAG blocks contain news + trade memory for each symbol.
        Macro context (global market data) is fetched once and appended at the
        end so it is not repeated N times in the combined prompt.
        The returned dict maps every symbol to either the chosen action or HOLD.
        """
        # Per-symbol sections (news + memory, no macro — see _build_rag_context_per_symbol)
        combined_rag = "\n\n".join(
            f"--- {sym} ---\n{rag}"
            for sym, rag in rag_per_symbol.items()
            if rag and rag != "=== NO CONTEXT AVAILABLE ==="
        ) or "=== NO CONTEXT AVAILABLE ==="

        # Append global macro context once instead of repeating it per symbol
        shared_macro = self.retriever.retrieve_macro("global macro crypto market conditions")
        if shared_macro:
            combined_rag = f"{combined_rag}\n\n{shared_macro}"

        system_prompt, user_message = self.builder.build(
            snapshots, analyses, patterns, combined_rag,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
            ml_context=ml_context,
        )

        decision = await self._decide(system_prompt, user_message, htf_analyses)

        # The LLM chose one symbol — all others default to HOLD
        decisions: dict[str, TradeDecision] = {}
        for sym in snapshots:
            if sym == decision.symbol:
                decisions[sym] = decision
            else:
                decisions[sym] = TradeDecision(
                    symbol=sym,
                    action=Action.HOLD,
                    reasoning=f"single_call:not_chosen (chose {decision.symbol})",
                    source=decision.source,
                )
        return decisions

    async def _decide(
        self,
        system_prompt: str,
        user_message: str,
        htf_analyses: dict[str, Any] | None = None,
    ) -> TradeDecision:
        decision = await self.llm.decide(system_prompt, user_message)
        if decision.action == Action.HOLD:
            return decision
        valid_symbols = set(getattr(self.settings, "trading_symbols", ["BTCUSDT", "ETHUSDT"]))
        if decision.symbol not in valid_symbols:
            fallback = next(iter(valid_symbols))
            return TradeDecision(symbol=fallback, action=Action.HOLD, reasoning="invalid_symbol", source="fallback_hold")

        # HTF confirmation gate: block entries when higher-timeframe trend disagrees
        htf_enabled = getattr(self.settings, "htf_confirmation_enabled", False)
        if htf_enabled and htf_analyses and decision.action in (Action.BUY, Action.SELL):
            htf = htf_analyses.get(decision.symbol)
            if htf is not None:
                htf_signal = htf.signal
                if decision.action == Action.BUY and htf_signal in (Signal.SELL, Signal.STRONG_SELL):
                    self.logger.warning(
                        "HTF confirmation failed: 1h=BUY but %s=%s — overriding to HOLD",
                        getattr(self.settings, "htf_timeframe", "4h"), htf_signal.value,
                    )
                    return TradeDecision(
                        symbol=decision.symbol, action=Action.HOLD,
                        reasoning=f"htf_disagreement:{htf_signal.value}", source="htf_filter",
                    )
                if decision.action == Action.SELL and htf_signal in (Signal.BUY, Signal.STRONG_BUY):
                    self.logger.warning(
                        "HTF confirmation failed: 1h=SELL but %s=%s — overriding to HOLD",
                        getattr(self.settings, "htf_timeframe", "4h"), htf_signal.value,
                    )
                    return TradeDecision(
                        symbol=decision.symbol, action=Action.HOLD,
                        reasoning=f"htf_disagreement:{htf_signal.value}", source="htf_filter",
                    )

        return decision
