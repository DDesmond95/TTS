from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("tts_platform.tasks.voice_clone")


class VoiceCloneRequest(BaseModel):
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
    def validate(self, request: VoiceCloneRequest) -> VoiceCloneRequest:
        return request

    async def run(self, engine: Any, request: VoiceCloneRequest) -> RunResult:
        model_id_or_path = engine._resolve_model(request.model, expected_kind="base")
        run_id, run_dir = engine.outputs.new_run_dir("voice_clone")

        ref_audio = request.ref_audio
        ref_text = request.ref_text
        x_vector_only_mode = request.x_vector_only_mode
        cached_prompt = None

        if request.voice_profile:
            prof = engine.voices.get(request.voice_profile)
            if not prof:
                raise ValueError(f"Voice profile not found: {request.voice_profile}")
            if prof.type != "clone" or not prof.clone:
                raise ValueError(f"Voice profile is not a clone profile: {request.voice_profile}")

            if not ref_audio:
                ref_audio = prof.clone.ref_audio_path
            if not ref_text and prof.clone.ref_text_path:
                ref_text = prof.clone.ref_text_path

            x_vector_only_mode = bool(prof.clone.x_vector_only_mode) if prof.clone.x_vector_only_mode is not None else x_vector_only_mode

            if request.use_cached_prompt and prof.clone and prof.clone.cached_prompt_path:
                import torch
                prompt_path = engine.voices.resolve_path(prof.clone.cached_prompt_path)
                if prompt_path.exists():
                    cached_prompt = torch.load(str(prompt_path), map_location="cpu", weights_only=False)

        ref_audio_resolved = engine.voices.resolve_path(ref_audio).as_posix() if ref_audio else None
        ref_text_str = None
        if ref_text and not x_vector_only_mode:
            rt = engine.voices.resolve_path(ref_text)
            if rt.exists():
                ref_text_str = rt.read_text(encoding="utf-8").strip()

        params = {
            "task": "voice_clone",
            "model": model_id_or_path,
            "text": request.text,
            "language": request.language,
            "voice_profile": request.voice_profile,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            "use_cached_prompt": request.use_cached_prompt,
            "gen": request.gen,
        }
        engine.outputs.write_params(run_dir, params)

        import asyncio
        async with engine._sem:
            model_obj = await asyncio.to_thread(
                engine._get_or_load, model_id_or_path, "base"
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
                    raise ValueError("ref_audio is required (or set voice_profile with ref_audio_path)")
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_clone,
                    text=request.text,
                    language=request.language,
                    ref_audio=ref_audio_resolved,
                    ref_text=ref_text_str,
                    x_vector_only_mode=x_vector_only_mode,
                    **request.gen,
                )

        audio_paths = []
        for i, w in enumerate(wavs):
            audio_paths.append(
                engine.outputs.save_wav(
                    run_dir, np.asarray(w), int(sr), filename=f"audio_{i}.wav"
                )
            )

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        engine.outputs.write_meta(run_dir, meta)

        audio_path = None
        if len(audio_paths) == 1:
            audio_path = audio_paths[0]
            if audio_path.name != "audio.wav":
                (run_dir / "audio.wav").write_bytes(audio_path.read_bytes())
                audio_path = (run_dir / "audio.wav").resolve()

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    async def stream(self, engine: Any, request: VoiceCloneRequest) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Streaming via sentence chunking."""
        if isinstance(request.text, list):
             yield (np.zeros(0), 0)
             return

        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', request.text) if s.strip()]
        if not sentences:
            return

        model_id_or_path = engine._resolve_model(request.model, expected_kind="base")

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
                x_vector_only_mode = bool(prof.clone.x_vector_only_mode) if prof.clone.x_vector_only_mode is not None else x_vector_only_mode
                if request.use_cached_prompt and prof.clone.cached_prompt_path:
                    import torch
                    prompt_path = engine.voices.resolve_path(prof.clone.cached_prompt_path)
                    if prompt_path.exists():
                        cached_prompt = torch.load(str(prompt_path), map_location="cpu", weights_only=False)

        ref_audio_resolved = engine.voices.resolve_path(ref_audio).as_posix() if ref_audio else None
        ref_text_str = None
        if ref_text and not x_vector_only_mode:
            rt = engine.voices.resolve_path(ref_text)
            if rt.exists():
                ref_text_str = rt.read_text(encoding="utf-8").strip()

        import asyncio
        model_obj = await asyncio.to_thread(
            engine._get_or_load, model_id_or_path, "base"
        )

        for sent in sentences:
            async with engine._sem:
                if cached_prompt is not None:
                    wavs, sr = await asyncio.to_thread(
                        model_obj.generate_voice_clone,
                        text=sent,
                        language=request.language,
                        voice_clone_prompt=cached_prompt,
                        **request.gen,
                    )
                else:
                    wavs, sr = await asyncio.to_thread(
                        model_obj.generate_voice_clone,
                        text=sent,
                        language=request.language,
                        ref_audio=ref_audio_resolved,
                        ref_text=ref_text_str,
                        x_vector_only_mode=x_vector_only_mode,
                        **request.gen,
                    )
                if wavs:
                    yield np.asarray(wavs[0]), int(sr)
