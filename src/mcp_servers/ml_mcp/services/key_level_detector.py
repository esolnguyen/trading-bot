"""A3: Key Level Detector.

Loads the DBSCAN clustering output from models/key_levels_cache.json
(produced by scripts/fit_key_levels.py) and formats nearby S/R levels
for the LLM prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from .model_store import load_json

logger = logging.getLogger(__name__)
_MAX_LEVELS = 4


class KeyLevelDetector:
    """Surface major historical S/R levels from a cached DBSCAN result."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._fallback: dict[str, Any] | None = load_json("1d/key_levels_cache")
        self._caches: dict[str, Any] = {}
        for sym in symbols or []:
            cache = load_json(f"1d/key_levels_{sym.lower()}_cache")
            if cache is not None:
                self._caches[sym.upper()] = cache
        self._cache = self._fallback

    def reload(self) -> None:
        """Refresh all caches from disk (call daily)."""
        self._fallback = load_json("1d/key_levels_cache")
        self._cache = self._fallback
        for sym in list(self._caches.keys()):
            cache = load_json(f"1d/key_levels_{sym.lower()}_cache")
            if cache is not None:
                self._caches[sym] = cache

    def _get_cache(self, symbol: str | None) -> dict[str, Any] | None:
        if symbol:
            return self._caches.get(symbol.upper(), self._fallback)
        return self._fallback

    def get_levels(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return the raw level list (copy) for structured consumers (e.g. MCP)."""
        cache = self._get_cache(symbol)
        if not cache:
            return []
        return list(cache.get("levels") or [])

    def format_context(
        self, current_price: float, symbol: str | None = None
    ) -> str | None:
        cache = self._get_cache(symbol)
        if not cache:
            return None

        levels: list[dict[str, Any]] = cache.get("levels", [])
        if not levels:
            return None

        # Sort by proximity to current price
        levels_sorted = sorted(levels, key=lambda l: abs(l["center"] - current_price))

        lines = ["## Major Historical S/R Levels (6-month clusters)"]
        shown = 0
        for lvl in levels_sorted:
            if shown >= _MAX_LEVELS:
                break
            center = lvl["center"]
            low_p = lvl["low"]
            high_p = lvl["high"]
            touches = lvl["touches"]
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
