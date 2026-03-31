"""A3: Key Level Detector.

Loads the DBSCAN clustering output from models/key_levels_cache.json
(produced by scripts/fit_key_levels.py) and formats nearby S/R levels
for the LLM prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from src.infrastructure.ml.model_store import load_json

logger = logging.getLogger(__name__)
_MAX_LEVELS = 4


class KeyLevelDetector:
    """Surface major historical S/R levels from a cached DBSCAN result."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] | None = load_json("key_levels_cache")

    def reload(self) -> None:
        """Refresh cache from disk (call daily)."""
        self._cache = load_json("key_levels_cache")

    def format_context(self, current_price: float) -> str | None:
        if not self._cache:
            return None

        levels: list[dict[str, Any]] = self._cache.get("levels", [])
        if not levels:
            return None

        # Sort by proximity to current price
        levels_sorted = sorted(levels, key=lambda l: abs(l["center"] - current_price))

        lines = ["## Major Historical S/R Levels (6-month clusters)"]
        shown = 0
        for lvl in levels_sorted:
            if shown >= _MAX_LEVELS:
                break
            center   = lvl["center"]
            low_p    = lvl["low"]
            high_p   = lvl["high"]
            touches  = lvl["touches"]
            lvl_type = lvl["type"]
            dist_pct = (center - current_price) / current_price * 100
            proximity = "← near current price" if abs(dist_pct) < 1.5 else ""
            lines.append(
                f"- {lvl_type.capitalize()}: ${low_p:,.0f}–${high_p:,.0f} "
                f"({touches} touches, {dist_pct:+.1f}%) {proximity}"
            )
            shown += 1

        if shown == 0:
            return None
        return "\n".join(lines)
