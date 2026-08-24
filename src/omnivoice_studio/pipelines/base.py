"""Base utilities for synthesis pipelines in OmniVoice Studio."""

from __future__ import annotations

import re
from typing import Any

import soundfile as sf
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask


class BasePipelineRequest(BaseModel):
    """Common request schema for pipelines."""

    task_type: str = "custom_voice"
    speaker: str = "Ryan"
    language: str = "Auto"
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class PipelineHelper:
    """Helper methods for pipelines to avoid code duplication."""

    @staticmethod
    def chunk_text(text: str, max_chars: int) -> list[str]:
        """Simple sentence-based chunking."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
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

    @staticmethod
    def finalize_run(
        engine: Any,
        *,
        run_name: str,
        combined_wav: Any,
        sample_rate: int,
        params: dict[str, Any],
        meta_extra: dict[str, Any] | None = None,
    ) -> RunResult:
        """Helper to save artifacts and finalize a run."""
        run_id, run_dir = engine.outputs.new_run_dir(run_name)
        audio_path = engine.outputs.save_wav(
            run_dir, combined_wav, sample_rate, filename="audio.wav"
        )
        engine.outputs.write_params(run_dir, params)

        meta = {
            "sample_rate": sample_rate,
        }
        if meta_extra:
            meta.update(meta_extra)

        engine.outputs.write_meta(run_dir, meta)

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=sample_rate,
            meta=meta,
        )

    @staticmethod
    async def run_task_and_load(
        engine: Any, task: Any, request: Any
    ) -> tuple[Any, int]:
        """Runs a task and loads its audio output into memory."""
        res = await task.run(engine, request)
        if not res.audio_path:
            raise RuntimeError("Task produced no audio file")
        wav, sr = sf.read(str(res.audio_path))
        return wav, sr

    @staticmethod
    def prepare_task_from_config(
        speaker_config: dict[str, Any],
        text: str,
        global_model: str | None,
        global_gen: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Prepares a task and its request based on a speaker configuration."""
        task_type = speaker_config.get("type", "custom_voice")
        model = global_model or speaker_config.get("model")
        lang = speaker_config.get("language", "Auto")

        if task_type == "custom_voice":

            task = CustomVoiceTask()
            task_req = CustomVoiceRequest(
                text=text,
                language=lang,
                speaker=speaker_config.get("speaker", "Ryan"),
                instruct=speaker_config.get("instruct", ""),
                model=model,
                gen=global_gen,
            )
        else:

            task = VoiceCloneTask()
            task_req = VoiceCloneRequest(
                text=text,
                language=lang,
                voice_profile=speaker_config.get("voice_profile"),
                ref_audio=speaker_config.get("ref_audio"),
                ref_text=speaker_config.get("ref_text"),
                model=model,
                gen=global_gen,
            )
        return task, task_req
