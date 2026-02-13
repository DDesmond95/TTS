from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("tts_platform.tasks.custom_voice")


class CustomVoiceRequest(BaseModel):
    text: str | list[str]
    language: str | list[str] = "Auto"
    speaker: str | list[str] = "Ryan"
    instruct: str | list[str] = ""
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class CustomVoiceTask(Task[CustomVoiceRequest, Any]):
    def validate(self, request: CustomVoiceRequest) -> CustomVoiceRequest:
        # Basic validation if needed
        return request

    async def run(self, engine: Any, request: CustomVoiceRequest) -> RunResult:
        model_id_or_path = engine._resolve_model(request.model, expected_kind="customvoice")
        params = {
            "task": "custom_voice",
            "model": model_id_or_path,
            "text": request.text,
            "language": request.language,
            "speaker": request.speaker,
            "instruct": request.instruct,
            "gen": request.gen,
        }
        run_id, run_dir = engine.outputs.new_run_dir("custom_voice")
        engine.outputs.write_params(run_dir, params)

        import asyncio
        async with engine._sem:
            model_obj = await asyncio.to_thread(
                engine._get_or_load, model_id_or_path, "customvoice"
            )
            wavs, sr = await asyncio.to_thread(
                model_obj.generate_custom_voice,
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                instruct=request.instruct,
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

    async def stream(self, engine: Any, request: CustomVoiceRequest) -> AsyncIterator[tuple[np.ndarray, int]]:
        """True streaming via sentence chunking."""
        if isinstance(request.text, list):
             # For batch, just run normally and yield each (backward compatibility or simple yield)
             await self.run(engine, request)
             # This is a bit complex since run() returns RunResult.
             # For now, let's focus on single string streaming.
             yield (np.zeros(0), 0) # Placeholder for batch
             return

        import re
        # Tokenize by sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', request.text) if s.strip()]
        if not sentences:
            return

        model_id_or_path = engine._resolve_model(request.model, expected_kind="customvoice")

        import asyncio
        model_obj = await asyncio.to_thread(
            engine._get_or_load, model_id_or_path, "customvoice"
        )

        for sent in sentences:
            async with engine._sem:
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_custom_voice,
                    text=sent,
                    language=request.language,
                    speaker=request.speaker,
                    instruct=request.instruct,
                    **request.gen,
                )
                if wavs:
                    yield np.asarray(wavs[0]), int(sr)
