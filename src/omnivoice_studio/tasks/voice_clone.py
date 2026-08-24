"""Task implementation for Qwen3-TTS Voice Clone generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.voice_clone")


class VoiceCloneRequest(BaseModel):
    """Request schema for voice clone generation."""

    text: str | list[str]
    language: str | list[str] = "Auto"
    ref_audio: str | None = None
    ref_text: str | None = None
    voice_profile: str | None = None
    model: str | None = None
    x_vector_only_mode: bool = False
    use_cached_prompt: bool = True
    gen: dict[str, Any] = Field(default_factory=dict)


class VoiceCloneTask(Task[VoiceCloneRequest, RunResult]):
    """Generates audio using Qwen3-TTS in 'voice_clone' mode."""

    def validate(self, request: VoiceCloneRequest) -> VoiceCloneRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: VoiceCloneRequest) -> RunResult:
        """Executes a voice cloning process."""
        model_id_or_path = engine.resolve_model(request.model, expected_kind="base")
        params = {
            "task": "voice_clone",
            "model": model_id_or_path,
            "text": request.text,
            "language": request.language,
            "voice_profile": request.voice_profile,
            "ref_audio": request.ref_audio,
            "ref_text": request.ref_text,
            "x_vector_only_mode": request.x_vector_only_mode,
            "use_cached_prompt": request.use_cached_prompt,
            "gen": request.gen,
        }
        run_id, run_dir = self._prepare_run(engine, "voice_clone", params)

        ref_audio = request.ref_audio
        ref_text = request.ref_text
        x_vector_only_mode = request.x_vector_only_mode
        cached_prompt = None

        # ... logic continues ...
        if request.voice_profile:
            prof = engine.voices.get(request.voice_profile)
            if not prof:
                raise ValueError(f"Voice profile not found: {request.voice_profile}")
            if prof.type != "clone" or not prof.clone:
                raise ValueError(
                    f"Voice profile is not a clone profile: {request.voice_profile}"
                )

            if not ref_audio:
                ref_audio = prof.clone.ref_audio_path
            if not ref_text and prof.clone.ref_text_path:
                ref_text = prof.clone.ref_text_path

            x_vector_only_mode = (
                bool(prof.clone.x_vector_only_mode)
                if prof.clone.x_vector_only_mode is not None
                else x_vector_only_mode
            )

            if (
                request.use_cached_prompt
                and prof.clone
                and prof.clone.cached_prompt_path
            ):
                prompt_path = engine.voices.resolve_path(prof.clone.cached_prompt_path)
                if prompt_path.exists():
                    cached_prompt = torch.load(
                        str(prompt_path), map_location="cpu", weights_only=False
                    )

        ref_audio_resolved = (
            engine.voices.resolve_path(ref_audio).as_posix() if ref_audio else None
        )
        ref_text_str = None
        if ref_text and not x_vector_only_mode:
            rt = engine.voices.resolve_path(ref_text)
            if rt.exists():
                ref_text_str = rt.read_text(encoding="utf-8").strip()

        async with engine.sem:
            model_obj = await asyncio.to_thread(
                engine.get_or_load, model_id_or_path, "base"
            )

            if cached_prompt is not None:
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_clone,
                    text=request.text,
                    language=request.language,
                    voice_clone_prompt=cached_prompt,
                    **request.gen,
                )
            else:
                if not ref_audio_resolved:
                    raise ValueError(
                        "ref_audio is required (or set voice_profile with ref_audio_path)"
                    )
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_clone,
                    text=request.text,
                    language=request.language,
                    ref_audio=ref_audio_resolved,
                    ref_text=ref_text_str,
                    x_vector_only_mode=x_vector_only_mode,
                    **request.gen,
                )

        return engine.outputs.complete_run(run_id, run_dir, wavs, sr)

    async def stream(
        self, engine: Any, request: VoiceCloneRequest
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Streaming for voice cloning (sentence-by-sentence)."""
        clone_s = self._get_stream_sentences(request.text)
        if not clone_s:
            return

        model_id_or_path = engine.resolve_model(request.model, expected_kind="base")

        # Resolve prompt info once
        ref_audio = request.ref_audio
        ref_text = request.ref_text
        x_vector_only_mode = request.x_vector_only_mode
        cached_prompt = None

        if request.voice_profile:
            prof = engine.voices.get(request.voice_profile)
            if prof and prof.type == "clone" and prof.clone:
                if not ref_audio:
                    ref_audio = prof.clone.ref_audio_path
                if not ref_text and prof.clone.ref_text_path:
                    ref_text = prof.clone.ref_text_path
                x_vector_only_mode = (
                    bool(prof.clone.x_vector_only_mode)
                    if prof.clone.x_vector_only_mode is not None
                    else x_vector_only_mode
                )
                if request.use_cached_prompt and prof.clone.cached_prompt_path:
                    prompt_path = engine.voices.resolve_path(
                        prof.clone.cached_prompt_path
                    )
                    if prompt_path.exists():
                        cached_prompt = torch.load(
                            str(prompt_path), map_location="cpu", weights_only=False
                        )

        ref_audio_resolved = (
            engine.voices.resolve_path(ref_audio).as_posix() if ref_audio else None
        )
        ref_text_str = None
        if ref_text and not x_vector_only_mode:
            rt = engine.voices.resolve_path(ref_text)
            if rt.exists():
                ref_text_str = rt.read_text(encoding="utf-8").strip()

        clone_model = await asyncio.to_thread(
            engine.get_or_load, model_id_or_path, "base"
        )

        if cached_prompt is not None:
            base_p = {"language": request.language, "voice_clone_prompt": cached_prompt}
        else:
            base_p = {
                "language": request.language,
                "ref_audio": ref_audio_resolved,
                "ref_text": ref_text_str,
                "x_vector_only_mode": x_vector_only_mode,
            }

        async for chunk in self._stream_loop(
            engine=engine,
            sentences=clone_s,
            model_obj=clone_model,
            method_name="generate_voice_clone",
            base_params=base_p,
            gen_params=request.gen,
        ):
            yield chunk
