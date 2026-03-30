"""FastAPI dashboard server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

_STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    """FastAPI-based dashboard server with WebSocket support."""

    def __init__(
        self,
        *,
        state: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        logger: Optional[logging.Logger] = None,
        # Optional rich-context dependencies injected from build_runtime
        brain_service: Optional[Any] = None,
        vector_memory: Optional[Any] = None,
        analysis_engine: Optional[Any] = None,
        persistence: Optional[Any] = None,
        unified_parser: Optional[Any] = None,
        exchange_manager: Optional[Any] = None,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger(__name__)
        self.brain_service = brain_service
        self.vector_memory = vector_memory
        self.analysis_engine = analysis_engine
        self.persistence = persistence
        self.unified_parser = unified_parser
        self.exchange_manager = exchange_manager
        self._server: Any = None
        self._app: Any = None

    def _build_app(self) -> Any:
        try:
            from fastapi import FastAPI  # noqa: PLC0415
            from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415
            from fastapi.staticfiles import StaticFiles  # noqa: PLC0415
            from fastapi.responses import HTMLResponse  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("fastapi is required for the dashboard") from exc

        app = FastAPI(title="Trading Bot Dashboard", version="1.0.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Inject state into request context via app.state
        app.state.dashboard = self.state
        app.state.brain_service = self.brain_service
        app.state.vector_memory = self.vector_memory
        app.state.analysis_engine = self.analysis_engine
        app.state.persistence = self.persistence
        app.state.unified_parser = self.unified_parser

        # Register routers
        try:
            from src.dashboard.routers.ws_router import router as ws_router  # noqa: PLC0415
            from src.dashboard.routers.monitor import router as monitor_router  # noqa: PLC0415
            from src.dashboard.routers.performance import router as perf_router  # noqa: PLC0415
            from src.dashboard.routers.brain import router as brain_router  # noqa: PLC0415
            from src.dashboard.routers.visuals import router as visuals_router  # noqa: PLC0415
            app.include_router(ws_router)
            app.include_router(monitor_router)
            app.include_router(perf_router)
            app.include_router(brain_router)
            app.include_router(visuals_router)
        except ImportError:
            self.logger.warning("Some dashboard routers could not be loaded")

        # Static files + SPA fallback
        if _STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def root() -> HTMLResponse:
            index = _STATIC_DIR / "index.html"
            if index.exists():
                return HTMLResponse(index.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Dashboard</h1>")

        return app

    async def start(self) -> None:
        """Start the uvicorn server (blocks until stopped)."""
        try:
            import uvicorn  # noqa: PLC0415
        except ImportError:
            self.logger.error("uvicorn is required for the dashboard")
            return

        self._app = self._build_app()
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self.logger.info("Dashboard starting on http://%s:%s", self.host, self.port)
        await self._server.serve()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

    def close(self) -> None:
        """Synchronous close hook for _close_runtime()."""
        if self._server is not None:
            self._server.should_exit = True
