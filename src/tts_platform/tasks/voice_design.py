from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("tts_platform.tasks.voice_design")


class VoiceDesignRequest(BaseModel):
    text: str | list[str]
    language: str | list[str] = "Auto"
    instruct: str | list[str] = ""
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class VoiceDesignTask(Task[VoiceDesignRequest, RunResult]):
    def validate(self, request: VoiceDesignRequest) -> VoiceDesignRequest:
        return request

    async def run(self, engine: Any, request: VoiceDesignRequest) -> RunResult:
        model_id_or_path = engine._resolve_model(request.model, expected_kind="voicedesign")
        params = {
            "task": "voice_design",
            "model": model_id_or_path,
            "text": request.text,
            "language": request.language,
            "instruct": request.instruct,
            "gen": request.gen,
        }
        run_id, run_dir = engine.outputs.new_run_dir("voice_design")
        engine.outputs.write_params(run_dir, params)

        import asyncio
        async with engine._sem:
            model_obj = await asyncio.to_thread(
                engine._get_or_load, model_id_or_path, "voicedesign"
            )
            wavs, sr = await asyncio.to_thread(
                model_obj.generate_voice_design,
                text=request.text,
                language=request.language,
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

    async def stream(self, engine: Any, request: VoiceDesignRequest) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Streaming via sentence chunking."""
        if isinstance(request.text, list):
             yield (np.zeros(0), 0)
             return

        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', request.text) if s.strip()]
        if not sentences:
            return

        model_id_or_path = engine._resolve_model(request.model, expected_kind="voicedesign")

        import asyncio
        model_obj = await asyncio.to_thread(
            engine._get_or_load, model_id_or_path, "voicedesign"
        )

        for sent in sentences:
            async with engine._sem:
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_design,
                    text=sent,
                    language=request.language,
                    instruct=request.instruct,
                    **request.gen,
                )
                if wavs:
                    yield np.asarray(wavs[0]), int(sr)
