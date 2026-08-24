"""Audiobook generation pipeline for OmniVoice Studio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..storage.outputs import RunResult
from .base import BasePipelineRequest, PipelineHelper
from .long_form import LongFormPipeline, LongFormRequest

log = logging.getLogger("omnivoice_studio.pipelines.audiobook")


class AudiobookRequest(BasePipelineRequest):
    """Configuration for an audiobook generation run."""

    chapter_paths: list[str]
    merge_all: bool = True


class AudiobookPipeline:
    """Pipeline for generating audiobooks by processing multiple chapter files."""

    async def run(self, engine: Any, request: AudiobookRequest) -> RunResult:
        """
        Executes the audiobook pipeline.

        Args:
            engine: The TTSEngine instance.
            request: The AudiobookRequest configuration.

        Returns:
            A RunResult object.
        """
        long_form = LongFormPipeline()
        chapter_results = []

        for i, path in enumerate(request.chapter_paths):
            p = Path(path)
            if not p.exists():
                log.warning("Chapter file not found: %s", path)
                continue

            text = p.read_text(encoding="utf-8")
            log.info("Processing chapter %d: %s", i + 1, p.name)

            # Reuse LongFormPipeline logic for each chapter
            lf_req = LongFormRequest(
                text=text,
                task_type=request.task_type,
                speaker=request.speaker,
                language=request.language,
                model=request.model,
                gen=request.gen,
            )
            res = await long_form.run(engine, lf_req)
            chapter_results.append(res)

        # Merge all chapters if requested
        combined_wav = []
        sample_rate = 24000

        for res in chapter_results:
            wav, sr = sf.read(str(res.audio_path))
            sample_rate = sr
            combined_wav.append(wav)
            # Add small silence between chapters
            combined_wav.append(np.zeros(int(sr * 2.0)))

        # 3. Finalize
        meta_extra = {
            "chapters_processed": len(chapter_results),
            "chapter_ids": [r.run_id for r in chapter_results],
        }

        merged_wav = (
            np.concatenate(combined_wav)
            if (request.merge_all and combined_wav)
            else np.zeros(0)
        )

        return PipelineHelper.finalize_run(
            engine,
            run_name="audiobook",
            combined_wav=merged_wav,
            sample_rate=sample_rate,
            params=request.model_dump(),
            meta_extra=meta_extra,
        )
