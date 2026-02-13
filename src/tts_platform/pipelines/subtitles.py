from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask

log = logging.getLogger("tts_platform.pipelines.subtitles")


class Caption(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class SubtitlesRequest(BaseModel):
    srt_path: str
    task_type: str = "custom_voice"
    speaker: str = "Ryan"
    language: str = "Auto"
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)
    preserve_timing: bool = True


class SubtitlesPipeline:
    async def run(self, engine: Any, request: SubtitlesRequest) -> Any:
        # 1. Parse SRT
        captions = self._parse_srt(request.srt_path)
        log.info("Processing %d captions", len(captions))

        all_wavs = []
        sample_rate = 24000
        current_time_ms = 0

        for _i, cap in enumerate(captions):
            # Handle gap if preserving timing
            if request.preserve_timing and cap.start_ms > current_time_ms:
                silence_len = int(sample_rate * ((cap.start_ms - current_time_ms) / 1000.0))
                all_wavs.append(np.zeros(silence_len))
                current_time_ms = cap.start_ms

            task = CustomVoiceTask()
            task_req = CustomVoiceRequest(
                text=cap.text,
                language=request.language,
                speaker=request.speaker,
                model=request.model,
                gen=request.gen,
            )

            res = await task.run(engine, task_req)

            import soundfile as sf
            wav, sr = sf.read(str(res.audio_path))
            sample_rate = sr
            all_wavs.append(wav)

            duration_ms = (len(wav) / sr) * 1000
            current_time_ms += duration_ms

        # 2. Stitch
        combined = np.concatenate(all_wavs)

        # 3. Save
        run_id, run_dir = engine.outputs.new_run_dir("subtitles")
        audio_path = engine.outputs.save_wav(run_dir, combined, sample_rate, filename="audio.wav")

        engine.outputs.write_params(run_dir, request.model_dump())

        meta = {
            "sample_rate": sample_rate,
            "captions": len(captions),
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

    def _parse_srt(self, path: str) -> list[Caption]:
        import re
        content = Path(path).read_text(encoding="utf-8")
        blocks = re.split(r'\n\s*\n', content.strip())
        captions = []

        for block in blocks:
            lines = block.split('\n')
            if len(lines) < 3:
                continue

            time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if not time_match:
                continue

            def to_ms(t_str):
                h, m, s_ms = t_str.split(':')
                s, ms = s_ms.split(',')
                return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

            start = to_ms(time_match.group(1))
            end = to_ms(time_match.group(2))
            text = " ".join(lines[2:])
            captions.append(Caption(start_ms=start, end_ms=end, text=text))

        return sorted(captions, key=lambda x: x.start_ms)
