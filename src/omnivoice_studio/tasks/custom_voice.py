"""Task implementation for Qwen3-TTS Custom Voice generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..exceptions import InferenceError
from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.custom_voice")


class CustomVoiceRequest(BaseModel):
    """Request schema for custom voice generation."""

    text: str | list[str]
    language: str | list[str] = "Auto"
    speaker: str | list[str] = "Ryan"
    instruct: str | list[str] = ""
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class CustomVoiceTask(Task[CustomVoiceRequest, Any]):
    """Generates audio using Qwen3-TTS in 'custom_voice' mode."""

    def validate(self, request: CustomVoiceRequest) -> CustomVoiceRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: CustomVoiceRequest) -> RunResult:
        """Execute the custom voice generation task."""
        cv_id = engine.resolve_model(request.model, expected_kind="customvoice")
        params = {
            "task": "custom_voice",
            "model": cv_id,
            "text": request.text,
            "language": request.language,
            "speaker": request.speaker,
            "instruct": request.instruct,
            "gen": request.gen,
        }
        run_id, run_dir = self._prepare_run(engine, "custom_voice", params)

        async with engine.sem:
            model_obj = await asyncio.to_thread(
                engine.get_or_load, cv_id, "customvoice"
            )

            try:
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_custom_voice,
                    text=request.text,
                    language=request.language,
                    speaker=request.speaker,
                    instruct=request.instruct,
                    **request.gen,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                log.exception("Inference failed for custom voice")
                raise InferenceError(f"Generation failed: {e}") from e

            if not wavs:
                raise InferenceError("Model returned no audio data")

        return engine.outputs.complete_run(run_id, run_dir, wavs, sr)

    async def stream(
        self, engine: Any, request: CustomVoiceRequest
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Streaming generation via sentence chunking."""
        cv_s = self._get_stream_sentences(request.text)
        if not cv_s:
            return

        cv_id = engine.resolve_model(request.model, expected_kind="customvoice")

        model_obj = await asyncio.to_thread(engine.get_or_load, cv_id, "customvoice")

        async for chunk in self._stream_loop(
            engine=engine,
            sentences=cv_s,
            model_obj=model_obj,
            method_name="generate_custom_voice",
            base_params={
                "language": request.language,
                "speaker": request.speaker,
                "instruct": request.instruct,
            },
            gen_params=request.gen,
        ):
            yield chunk
