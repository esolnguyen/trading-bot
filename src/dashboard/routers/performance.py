"""Performance API endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

router = APIRouter(prefix="/api/performance") if _FASTAPI else None

if _FASTAPI:
    @router.get("/history")
    async def get_history(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"history": state.get("trade_history", []) if state else []})

    @router.get("/stats")
    async def get_stats(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"stats": state.get("statistics", {}) if state else {}})
