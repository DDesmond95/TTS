from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..config import AppConfig
from ..engine.engine import TTSEngine
from .http_routes import get_engine as http_get_engine
from .http_routes import router as http_router
from .ws_routes import get_engine as ws_get_engine
from .ws_routes import router as ws_router

log = logging.getLogger("tts_platform.api.app")


def create_app(cfg: AppConfig, repo_root: Path) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Local Platform", version="0.1.0")

    engine = TTSEngine(
        models_dir=Path(cfg.paths.models_dir),
        voices_dir=Path(cfg.paths.voices_dir),
        outputs_dir=Path(cfg.paths.outputs_dir),
        runtime=cfg.runtime,
    )

    # mount engine into state
    app.state.engine = engine

    # DI override
    def _engine_dep() -> TTSEngine:
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
            if request.url.path in ("/health", "/ready"):
                return await call_next(request)
            key = request.headers.get("x-api-key")
            if key != cfg.api.api_key:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.include_router(http_router)
    app.include_router(ws_router)

    return app
