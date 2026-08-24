"""FastAPI application factory for OmniVoice Studio."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import AppConfig
from ..engine.engine import TTSEngine
from ..exceptions import (
    AudioProcessingError,
    ModelLoadError,
    ModelNotFoundError,
    OmniVoiceError,
)
from .http_routes import get_engine as http_get_engine
from .http_routes import router as http_router
from .ws_routes import get_engine as ws_get_engine
from .ws_routes import router as ws_router

log = logging.getLogger("omnivoice_studio.api.app")


def create_app(cfg: AppConfig, repo_root: Path) -> FastAPI:
    """
    Creates and configures a FastAPI application instance.

    Args:
        cfg: The application configuration.
        repo_root: The root directory of the repository.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(title="OmniVoice Studio", version="0.1.0")

    @app.exception_handler(OmniVoiceError)
    async def omnivoice_exception_handler(_request: Request, exc: OmniVoiceError):
        """Global exception handler for OmniVoiceError."""
        status_code = 400
        if isinstance(exc, ModelNotFoundError):
            status_code = 404
        elif isinstance(exc, (ModelLoadError, AudioProcessingError)):
            status_code = 500

        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    engine = TTSEngine(
        models_dir=repo_root / cfg.paths.models_dir,
        voices_dir=repo_root / cfg.paths.voices_dir,
        outputs_dir=repo_root / cfg.paths.outputs_dir,
        runtime=cfg.runtime,
    )

    # mount engine into state
    app.state.engine = engine

    # DI override
    def _engine_dep() -> TTSEngine:
        """Dependency provider for TTSEngine instance."""
        return app.state.engine

    app.dependency_overrides[http_get_engine] = _engine_dep
    app.dependency_overrides[ws_get_engine] = _engine_dep

    if cfg.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # simple API key middleware (optional)
    if cfg.api.api_key:

        @app.middleware("http")
        async def api_key_guard(request: Request, call_next):
            """Middleware to protect routes with an API key."""
            if request.url.path in ("/health", "/ready"):
                return await call_next(request)
            key = request.headers.get("x-api-key")
            if key != cfg.api.api_key:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.include_router(http_router)
    app.include_router(ws_router)

    return app
