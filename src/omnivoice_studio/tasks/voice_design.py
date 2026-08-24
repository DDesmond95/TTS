"""Task implementation for Qwen3-TTS Voice Design generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.voice_design")


class VoiceDesignRequest(BaseModel):
    """Request schema for voice design generation."""

    text: str | list[str]
    language: str | list[str] = "Auto"
    instruct: str | list[str] = ""
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class VoiceDesignTask(Task[VoiceDesignRequest, RunResult]):
    """Generates audio using Qwen3-TTS in 'voice_design' mode."""

    def validate(self, request: VoiceDesignRequest) -> VoiceDesignRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: VoiceDesignRequest) -> RunResult:
        """Execute a voice design generation."""
        vd_id = engine.resolve_model(request.model, expected_kind="voicedesign")
        params = {
            "task": "voice_design",
            "model": vd_id,
            "text": request.text,
            "language": request.language,
            "speaker": "Custom",
            "instruct": request.instruct,
            "gen": request.gen,
        }
        run_id, run_dir = self._prepare_run(engine, "voice_design", params)

        async with engine.sem:
            model_obj = await asyncio.to_thread(
                engine.get_or_load, vd_id, "voicedesign"
            )
            wavs, sr = await asyncio.to_thread(
                model_obj.generate_voice_design,
                text=request.text,
                language=request.language,
                instruct=request.instruct,
                **request.gen,
            )

        return engine.outputs.complete_run(run_id, run_dir, wavs, sr)

    async def stream(
        self, engine: Any, request: VoiceDesignRequest
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Streaming for voice design (sentence-by-sentence)."""
        vd_s = self._get_stream_sentences(request.text)
        if not vd_s:
            return

        vd_id = engine.resolve_model(request.model, expected_kind="voicedesign")

        design_model = await asyncio.to_thread(engine.get_or_load, vd_id, "voicedesign")

        async for chunk in self._stream_loop(
            engine=engine,
            sentences=vd_s,
            model_obj=design_model,
            method_name="generate_voice_design",
            base_params={"language": request.language, "instruct": request.instruct},
            gen_params=request.gen,
        ):
            yield chunk
