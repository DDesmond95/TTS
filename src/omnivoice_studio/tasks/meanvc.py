"""Task implementation for MeanVC voice conversion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torchaudio
from pydantic import BaseModel, Field

from ..exceptions import AudioProcessingError, InferenceError
from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.meanvc")


class VoiceConversionRequest(BaseModel):
    """Request schema for MeanVC voice conversion."""

    source_audio: str
    target_speaker_audio: str | None = None
    target_speaker_id: str | None = None
    model: str | None = None
    steps: int = 5
    gen: dict[str, Any] = Field(default_factory=dict)


class VoiceConversionTask(Task[VoiceConversionRequest, RunResult]):
    """Voice conversion using MeanVC models."""

    def validate(self, request: VoiceConversionRequest) -> VoiceConversionRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: VoiceConversionRequest) -> RunResult:
        """Execute the voice conversion task."""
        model_id_or_path = engine.resolve_model(request.model, expected_kind="meanvc")

        run_id, run_dir = self._prepare_run(
            engine, "voice_conversion", request.model_dump()
        )

        async with engine.sem:
            _objs = await asyncio.to_thread(
                engine.get_or_load, model_id_or_path, "meanvc"
            )
            # _model = _objs["model"]
            # _vocos = _objs["vocos"]
            device = engine.runtime.device

            # Load source audio
            try:
                src_wav, _sr_in = torchaudio.load(request.source_audio)
                src_wav = src_wav.to(device)
            except (RuntimeError, ValueError, OSError) as e:
                raise AudioProcessingError(
                    f"Failed to load source audio {request.source_audio}: {e}"
                ) from e

            log.info("Running MeanVC conversion on %s", request.source_audio)

            try:
                # Placeholder for actual inference logic
                wav = np.zeros((16000,))
                sr_out = 24000
            except (RuntimeError, ValueError, TypeError) as e:
                raise InferenceError(f"MeanVC inference failed: {e}") from e

        return engine.outputs.complete_run(run_id, run_dir, [wav], sr_out)
