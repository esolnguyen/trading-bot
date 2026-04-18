"""Shared numba shim.

Re-exports ``numba.njit`` when available; otherwise provides a no-op decorator
that preserves the decorated function's signature. Every indicator module
should import ``njit`` from here rather than defining its own fallback.
"""

from __future__ import annotations

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError:
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """Fallback no-op decorator when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


__all__ = ["njit"]
