from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .long_form import LongFormPipeline, LongFormRequest

log = logging.getLogger("tts_platform.pipelines.audiobook")

class AudiobookRequest(BaseModel):
    chapter_paths: list[str]
    task_type: str = "custom_voice"
    speaker: str = "Ryan" # Default speaker for the narrator
    language: str = "Auto"
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)
    merge_all: bool = True

class AudiobookPipeline:
    async def run(self, engine: Any, request: AudiobookRequest) -> Any:
        long_form = LongFormPipeline()
        chapter_results = []

        for i, path in enumerate(request.chapter_paths):
            p = Path(path)
            if not p.exists():
                log.warning("Chapter file not found: %s", path)
                continue

            text = p.read_text(encoding="utf-8")
            log.info("Processing chapter %d: %s", i+1, p.name)

            # Reuse LongFormPipeline logic for each chapter
            lf_req = LongFormRequest(
                text=text,
                task_type=request.task_type,
                speaker=request.speaker,
                language=request.language,
                model=request.model,
                gen=request.gen
            )
            res = await long_form.run(engine, lf_req)
            chapter_results.append(res)

        # Merge all chapters if requested
        combined_wav = []
        sample_rate = 24000

        import soundfile as sf
        for res in chapter_results:
            wav, sr = sf.read(str(res.audio_path))
            sample_rate = sr
            combined_wav.append(wav)
            # Add small silence between chapters
            combined_wav.append(np.zeros(int(sr * 2.0)))

        run_id, run_dir = engine.outputs.new_run_dir("audiobook")

        if request.merge_all and combined_wav:
            merged = np.concatenate(combined_wav)
            audio_path = engine.outputs.save_wav(run_dir, merged, sample_rate, filename="full_book.wav")
        else:
            audio_path = None

        meta = {
            "chapters_processed": len(chapter_results),
            "sample_rate": sample_rate,
            "chapter_ids": [r.run_id for r in chapter_results]
        }
        engine.outputs.write_meta(run_dir, meta)
        engine.outputs.write_params(run_dir, request.model_dump())

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=sample_rate,
            meta=meta
        )
