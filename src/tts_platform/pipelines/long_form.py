from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask

log = logging.getLogger("tts_platform.pipelines.long_form")


class LongFormRequest(BaseModel):
    text: str
    task_type: str = "custom_voice"  # custom_voice | voice_clone
    language: str = "Auto"
    # settings for custom_voice
    speaker: str = "Ryan"
    instruct: str = ""
    # settings for voice_clone
    voice_profile: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    # common
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)
    # chunking
    max_chars_per_chunk: int = 500
    silence_padding_ms: int = 500


class LongFormPipeline:
    async def run(self, engine: Any, request: LongFormRequest) -> Any:
        # 1. Chunk text
        chunks = self._chunk_text(request.text, request.max_chars_per_chunk)
        log.info("Split text into %d chunks", len(chunks))

        all_wavs = []
        sample_rate = 24000  # Default to 24k, will be updated by first chunk

        for i, chunk in enumerate(chunks):
            log.info("Processing chunk %d/%d", i + 1, len(chunks))
            task: Any
            task_req: Any
            if request.task_type == "custom_voice":
                task = CustomVoiceTask()
                task_req = CustomVoiceRequest(
                    text=chunk,
                    language=request.language,
                    speaker=request.speaker,
                    instruct=request.instruct,
                    model=request.model,
                    gen=request.gen,
                )
            else:
                task = VoiceCloneTask()
                task_req = VoiceCloneRequest(
                    text=chunk,
                    language=request.language,
                    voice_profile=request.voice_profile,
                    ref_audio=request.ref_audio,
                    ref_text=request.ref_text,
                    model=request.model,
                    gen=request.gen,
                )

            res = await task.run(engine, task_req)

            # Load the audio for stitching
            import soundfile as sf
            wav, sr = sf.read(str(res.audio_path))
            sample_rate = sr
            all_wavs.append(wav)

            if request.silence_padding_ms > 0 and i < len(chunks) - 1:
                silence_len = int(sr * (request.silence_padding_ms / 1000.0))
                all_wavs.append(np.zeros(silence_len))

        # 2. Stitch
        combined = np.concatenate(all_wavs)

        # 3. Save
        run_id, run_dir = engine.outputs.new_run_dir("long_form")
        audio_path = engine.outputs.save_wav(run_dir, combined, sample_rate, filename="audio.wav")

        params = request.model_dump()
        engine.outputs.write_params(run_dir, params)

        meta = {
            "sample_rate": sample_rate,
            "chunks": len(chunks),
            "duration_sec": len(combined) / sample_rate
        }
        engine.outputs.write_meta(run_dir, meta)

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=sample_rate,
            meta=meta,
        )

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        # Simple sentence-based chunking
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            if len(current_chunk) + len(s) < max_chars:
                current_chunk += (" " if current_chunk else "") + s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = s
                # If a single sentence is too long, split it by force
                while len(current_chunk) > max_chars:
                    chunks.append(current_chunk[:max_chars])
                    current_chunk = current_chunk[max_chars:]

        if current_chunk:
            chunks.append(current_chunk)
        return chunks
