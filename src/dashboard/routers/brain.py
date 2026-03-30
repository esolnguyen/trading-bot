"""Brain activity API endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

router = APIRouter(prefix="/api/brain") if _FASTAPI else None


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call sync or async callable safely, returning None on failure."""
    import asyncio  # noqa: PLC0415
    try:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(result)
        return result
    except Exception:  # noqa: BLE001
        return None


if _FASTAPI:
    @router.get("/status")
    async def get_brain_status(request: Request) -> Any:
        brain = getattr(request.app.state, "brain_service", None)
        if brain is None:
            return JSONResponse({"enabled": False})
        stats = _safe_call(getattr(brain, "get_stats", lambda: {}))
        return JSONResponse({"enabled": True, "stats": stats or {}})

    @router.get("/memory")
    async def get_brain_memory(request: Request) -> Any:
        vm = getattr(request.app.state, "vector_memory", None)
        if vm is None:
            return JSONResponse({"memories": []})
        memories = _safe_call(getattr(vm, "get_recent_memories", lambda: []))
        return JSONResponse({"memories": memories or []})

    @router.get("/rules")
    async def get_brain_rules(request: Request) -> Any:
        vm = getattr(request.app.state, "vector_memory", None)
        if vm is None:
            return JSONResponse({"rules": []})
        rules = _safe_call(getattr(vm, "get_semantic_rules", lambda: []))
        return JSONResponse({"rules": rules or []})

    @router.get("/vectors")
    async def get_brain_vectors(request: Request) -> Any:
        vm = getattr(request.app.state, "vector_memory", None)
        if vm is None:
            return JSONResponse({"vectors": []})
        vectors = _safe_call(getattr(vm, "get_all_vectors", lambda: []))
        return JSONResponse({"vectors": vectors or []})

    @router.get("/position")
    async def get_position(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"position": state.get("current_position") if state else None})

    @router.get("/refresh-price")
    async def refresh_price(request: Request) -> Any:
        state = getattr(request.app.state, "dashboard", None)
        return JSONResponse({"price": state.get("current_price") if state else None})
