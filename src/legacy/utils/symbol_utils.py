"""Symbol normalization helpers.

Trading symbols arrive in multiple shapes (``BTC/USDT``, ``BTCUSDT``,
``BTC/USDT:USDT``). Persistence and routing code needs a single canonical form
when building filenames or dictionary keys.
"""

from __future__ import annotations


def slugify_symbol(symbol: str) -> str:
    """Return a lowercase alphanumeric slug suitable for filenames."""
    return symbol.replace("/", "").replace(":", "").lower()


__all__ = ["slugify_symbol"]
