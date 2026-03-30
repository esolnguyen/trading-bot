"""Monitor API endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

router = APIRouter(prefix="/api/monitor") if _FASTAPI else None

if _FASTAPI:
    @router.get("/status")
    async def get_status(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        if state is None:
            return JSONResponse({"status": "no state"})
        return JSONResponse({"status": "running", **{
            k: v for k, v in state.get_all().items()
            if k in ("cycle", "last_decision", "last_symbol")
        }})

    @router.get("/last-prompt")
    async def get_last_prompt(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"prompt": state.get("last_prompt") if state else None})

    @router.get("/last-response")
    async def get_last_response(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"response": state.get("last_response") if state else None})

    @router.get("/system-prompt")
    async def get_system_prompt(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"system_prompt": state.get("system_prompt") if state else None})

    @router.get("/costs")
    async def get_costs(request: Request) -> Any:
        try:
            from src.shared.token_counter import CostStorage  # noqa: PLC0415
            costs = CostStorage().load()
            return JSONResponse({"costs": costs})
        except Exception:  # noqa: BLE001
            return JSONResponse({"costs": {}})

    @router.get("/news")
    async def get_news(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"news": state.get("latest_news", []) if state else []})
