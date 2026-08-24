"""Task implementation for TCSinger singing synthesis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.tcsinger")


class SingingSynthesisRequest(BaseModel):
    """Request schema for singing synthesis."""

    lyrics: str
    score: str | None = None
    ref_audio: str | None = None
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class SingingSynthesisTask(Task[SingingSynthesisRequest, RunResult]):
    """Singing synthesis using TCSinger models."""

    def validate(self, request: SingingSynthesisRequest) -> SingingSynthesisRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: SingingSynthesisRequest) -> RunResult:
        """Execute the singing synthesis task."""
        _model_id_or_path = engine.resolve_model(
            request.model, expected_kind="tcsinger"
        )

        run_id, run_dir = self._prepare_run(
            engine, "singing_synthesis", request.model_dump()
        )

        async with engine.sem:
            # _sampler = await asyncio.to_thread(
            #     engine.get_or_load, _model_id_or_path, "tcsinger"
            # )

            log.info("Synthesizing singing for lyrics: %s...", request.lyrics[:50])

            # Placeholder for actual inference logic
            wav = np.zeros((48000 * 5,))  # 5 seconds of silence
            sr = 48000

        meta_extra = {"lyrics_len": len(request.lyrics)}
        return engine.outputs.complete_run(
            run_id, run_dir, [wav], sr, meta_extra=meta_extra
        )
