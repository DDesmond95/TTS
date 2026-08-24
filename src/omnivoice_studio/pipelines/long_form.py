"""Long-form text-to-speech pipeline for OmniVoice Studio."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..storage.outputs import RunResult
from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask
from .base import BasePipelineRequest, PipelineHelper

log = logging.getLogger("omnivoice_studio.pipelines.long_form")


class LongFormRequest(BasePipelineRequest):
    """Configuration for a long-form synthesis run."""

    text: str
    instruct: str = ""
    # settings for voice_clone
    voice_profile: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    # chunking
    max_chars_per_chunk: int = 500
    silence_padding_ms: int = 500


class LongFormPipeline:
    """Pipeline for processing large amounts of text by chunking and stitching."""

    async def run(self, engine: Any, request: LongFormRequest) -> RunResult:
        """
        Executes the long-form pipeline.

        Args:
            engine: The TTSEngine instance.
            request: The LongFormRequest configuration.

        Returns:
            A RunResult object containing the stitched audio.
        """
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

            wav, sr = await PipelineHelper.run_task_and_load(engine, task, task_req)
            sample_rate = sr
            all_wavs.append(wav)

            if request.silence_padding_ms > 0 and i < len(chunks) - 1:
                silence_len = int(sr * (request.silence_padding_ms / 1000.0))
                all_wavs.append(np.zeros(silence_len))

        # 2. Stitch
        combined = np.concatenate(all_wavs)

        # 3. Finalize
        meta_extra = {
            "chunks": len(chunks),
            "duration_sec": len(combined) / sample_rate,
        }
        return PipelineHelper.finalize_run(
            engine,
            run_name="long_form",
            combined_wav=combined,
            sample_rate=sample_rate,
            params=request.model_dump(),
            meta_extra=meta_extra,
        )

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        """Simple sentence-based chunking."""
        return PipelineHelper.chunk_text(text, max_chars)
