"""WebSocket router for real-time dashboard updates."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
    import json
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

logger = logging.getLogger(__name__)

router = APIRouter() if _FASTAPI else None


class ConnectionManager:
    """WebSocket connection manager with per-IP rate limiting."""

    def __init__(self, max_per_ip: int = 10) -> None:
        self.active: List[Any] = []
        self._ip_counts: Dict[str, int] = defaultdict(int)
        self.max_per_ip = max_per_ip

    async def connect(self, ws: Any, client_ip: str) -> bool:
        if self._ip_counts[client_ip] >= self.max_per_ip:
            await ws.close(code=1008)
            return False
        await ws.accept()
        self.active.append(ws)
        self._ip_counts[client_ip] += 1
        return True

    def disconnect(self, ws: Any, client_ip: str) -> None:
        if ws in self.active:
            self.active.remove(ws)
        if self._ip_counts[client_ip] > 0:
            self._ip_counts[client_ip] -= 1

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = []
        data = json.dumps(payload)
        for ws in list(self.active):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


manager = ConnectionManager()


async def broadcast(payload: Dict[str, Any]) -> None:
    """Module-level broadcast function registered with DashboardState."""
    await manager.broadcast(payload)


if _FASTAPI:
    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        client_ip = ws.client.host if ws.client else "unknown"
        connected = await manager.connect(ws, client_ip)
        if not connected:
            return
        try:
            # Send initial snapshot
            state = getattr(ws.app.state, "dashboard", None)
            if state is not None:
                snapshot = state.get_all()
                if snapshot:
                    await ws.send_text(json.dumps(snapshot))
                state.register_broadcast(broadcast)
            while True:
                await asyncio.sleep(30)
                await ws.send_text(json.dumps({"ping": True}))
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(ws, client_ip)
