from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask

log = logging.getLogger("tts_platform.pipelines.script_read")


class ScriptRow(BaseModel):
    speaker: str
    text: str


class ScriptReadRequest(BaseModel):
    script_text: str  # Format: "Speaker: Text"
    speaker_map: dict[str, dict[str, Any]]  # Map "Speaker" tag to task parameters
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)
    silence_padding_ms: int = 800


class ScriptReadPipeline:
    async def run(self, engine: Any, request: ScriptReadRequest) -> Any:
        # 1. Parse script
        rows = self._parse_script(request.script_text)
        log.info("Parsed %d lines from script", len(rows))

        all_wavs = []
        sample_rate = 24000

        for i, row in enumerate(rows):
            task: Any
            task_req: Any
            speaker_config = request.speaker_map.get(row.speaker)
            if not speaker_config:
                log.warning("Unknown speaker tag: %s, skipping", row.speaker)
                continue

            task_type = speaker_config.get("type", "custom_voice")

            if task_type == "custom_voice":
                task = CustomVoiceTask()
                task_req = CustomVoiceRequest(
                    text=row.text,
                    language=speaker_config.get("language", "Auto"),
                    speaker=speaker_config.get("speaker", "Ryan"),
                    instruct=speaker_config.get("instruct", ""),
                    model=request.model or speaker_config.get("model"),
                    gen=request.gen,
                )
            else:
                task = VoiceCloneTask()
                task_req = VoiceCloneRequest(
                    text=row.text,
                    language=speaker_config.get("language", "Auto"),
                    voice_profile=speaker_config.get("voice_profile"),
                    ref_audio=speaker_config.get("ref_audio"),
                    ref_text=speaker_config.get("ref_text"),
                    model=request.model or speaker_config.get("model"),
                    gen=request.gen,
                )

            res = await task.run(engine, task_req)

            import soundfile as sf
            wav, sr = sf.read(str(res.audio_path))
            sample_rate = sr
            all_wavs.append(wav)

            if request.silence_padding_ms > 0 and i < len(rows) - 1:
                silence_len = int(sr * (request.silence_padding_ms / 1000.0))
                all_wavs.append(np.zeros(silence_len))

        # 2. Stitch
        combined = np.concatenate(all_wavs)

        # 3. Save
        run_id, run_dir = engine.outputs.new_run_dir("script_read")
        audio_path = engine.outputs.save_wav(run_dir, combined, sample_rate, filename="audio.wav")

        engine.outputs.write_params(run_dir, request.model_dump())

        meta = {
            "sample_rate": sample_rate,
            "lines": len(rows),
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

    def _parse_script(self, text: str) -> list[ScriptRow]:
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            match = re.match(r"^([^:]+):\s*(.*)$", line)
            if match:
                rows.append(ScriptRow(speaker=match.group(1).strip(), text=match.group(2).strip()))
        return rows
