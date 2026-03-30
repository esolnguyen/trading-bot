"""Dashboard state container."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional


class DashboardState:
    """Shared mutable state for the dashboard, updated by the trading loop."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = {}
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._ws_broadcast: Optional[Callable] = None

    # --- Core data access ---

    async def update(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = value
        await self._broadcast({key: value})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._data)

    # --- TTL cache ---

    def get_cached(self, key: str, ttl: float = 60.0) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > ttl:
            del self._cache[key]
            return None
        return value

    def set_cached(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.monotonic())

    def invalidate_cache(self, key: str) -> None:
        self._cache.pop(key, None)

    # --- WebSocket broadcast ---

    def register_broadcast(self, broadcast_fn: Callable) -> None:
        self._ws_broadcast = broadcast_fn

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if self._ws_broadcast is not None:
            try:
                await self._ws_broadcast(payload)
            except Exception:  # noqa: BLE001
                pass
