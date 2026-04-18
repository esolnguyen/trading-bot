"""Reflection routines — synthesize semantic rules from wins and losses."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List


def count_patterns(experiences: List[Any], key_builder: Callable[[dict], str]) -> Counter:
    """Count patterns in experiences by a caller-supplied key function."""
    return Counter(key_builder(exp.metadata) for exp in experiences)


def run_win_reflection(vector_memory: Any, logger: logging.Logger) -> None:
    """Reflect on recent winning trades and synthesize a semantic rule.

    Runs every N trades (scheduled by the caller).  Only stores a rule when
    the dominant winning pattern occurs ≥5 times and has win rate ≥60 %.
    """
    try:
        experiences = vector_memory.retrieve_similar_experiences(
            "recent trading experiences",
            k=20,
            use_decay=True,
            where={"outcome": "WIN"},
        )

        wins = experiences
        if len(wins) < 10:
            logger.debug("Not enough winning trades for reflection (need 10+)")
            return

        def build_win_key(meta: dict) -> str:
            regime = meta.get("market_regime", "NEUTRAL")
            adx = meta.get("adx_at_entry", 0)
            direction = meta.get("direction", "UNKNOWN")
            adx_label = (
                "HIGH_ADX" if adx >= 25 else "LOW_ADX" if adx < 20 else "MED_ADX"
            )
            return f"{direction}|{regime}|{adx_label}"

        pattern_counts = count_patterns(wins, build_win_key)
        if not pattern_counts:
            return

        pattern_key, count = pattern_counts.most_common(1)[0]
        if count < 5:
            logger.debug(
                "Pattern %s rejected: only %s occurrences (need 5+)",
                pattern_key, count,
            )
            return

        all_pattern_experiences = vector_memory.retrieve_similar_experiences(
            "recent trading experiences", k=50, use_decay=True
        )
        pattern_wins = sum(
            1 for exp in all_pattern_experiences
            if exp.metadata.get("outcome") == "WIN"
            and build_win_key(exp.metadata) == pattern_key
        )
        pattern_total = sum(
            1 for exp in all_pattern_experiences
            if build_win_key(exp.metadata) == pattern_key
        )

        if pattern_total > 0:
            win_rate = pattern_wins / pattern_total
            if win_rate < 0.6:
                logger.debug(
                    "Pattern %s rejected: win rate %s < 60%% (%s/%s trades)",
                    pattern_key, f"{win_rate:.0%}", pattern_wins, pattern_total,
                )
                return

        parts = pattern_key.split("|")
        direction = parts[0]
        regime = parts[1]
        adx_level = parts[2].replace("_", " ").title()

        rule_text = (
            f"{direction} trades perform well in {regime} market with {adx_level}. "
            f"({count} recent wins follow this pattern)"
        )

        rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vector_memory.store_semantic_rule(
            rule_id=rule_id,
            rule_text=rule_text,
            metadata={
                "source_pattern": pattern_key,
                "source_win_count": count,
                "total_analyzed": len(wins),
            },
        )
        logger.info("Reflection complete: stored rule '%s'", rule_text)

    except Exception as e:
        logger.warning("Reflection failed: %s", e)


def run_loss_reflection(vector_memory: Any, logger: logging.Logger) -> None:
    """Reflect on recent losses and synthesize an anti-pattern rule.

    Runs every N trades.  Stores a rule when ≥3 losses share the same
    (direction, market_regime, close_reason) signature.
    """
    try:
        experiences = vector_memory.retrieve_similar_experiences(
            "recent trading experiences",
            k=20,
            use_decay=True,
            where={"outcome": "LOSS"},
        )

        losses = experiences
        if len(losses) < 5:
            logger.debug(
                "Not enough losing trades for anti-pattern reflection (need 5+)"
            )
            return

        def build_loss_key(meta: dict) -> str:
            regime = meta.get("market_regime", "NEUTRAL")
            close_reason = meta.get("close_reason", "unknown")
            direction = meta.get("direction", "UNKNOWN")
            return f"{direction}|{regime}|{close_reason}"

        pattern_counts: Dict[str, int] = count_patterns(losses, build_loss_key)
        if not pattern_counts:
            return

        pattern_key, count = pattern_counts.most_common(1)[0]
        if count < 3:
            logger.debug(
                "Anti-pattern %s rejected: only %s occurrences (need 3+)",
                pattern_key, count,
            )
            return

        parts = pattern_key.split("|")
        direction = parts[0]
        regime = parts[1]
        close_reason = parts[2]

        rule_text = (
            f"⚠️ AVOID: {direction} trades in {regime} market often hit {close_reason}. "
            f"({count} recent losses follow this pattern)"
        )

        rule_id = f"anti_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vector_memory.store_semantic_rule(
            rule_id=rule_id,
            rule_text=rule_text,
            metadata={
                "rule_type": "anti_pattern",
                "source_pattern": pattern_key,
                "source_loss_count": count,
            },
        )
        logger.info("Anti-pattern reflection complete: stored rule '%s'", rule_text)

    except Exception as e:
        logger.warning("Loss reflection failed: %s", e)
