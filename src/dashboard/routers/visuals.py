"""Visuals/chart API endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

router = APIRouter(prefix="/api/visuals") if _FASTAPI else None

if _FASTAPI:
    @router.get("/charts/latest")
    async def get_latest_charts(request: Request) -> Any:
        engine = getattr(request.app.state, "analysis_engine", None)
        charts: Any = {}
        if engine is not None:
            charts = getattr(engine, "latest_charts", None) or getattr(engine, "last_chart", None) or {}
        return JSONResponse({"charts": charts})
