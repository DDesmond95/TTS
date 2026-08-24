"""Script reading pipeline for dialogue-based multi-speaker synthesis."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import PipelineHelper

log = logging.getLogger("omnivoice_studio.pipelines.script_read")


class ScriptRow(BaseModel):
    """Represents a single line from a script with a speaker tag."""

    speaker: str
    text: str


class ScriptReadRequest(BaseModel):
    """Configuration for a script reading synthesis run."""

    script_text: str  # Format: "Speaker: Text"
    speaker_map: dict[str, dict[str, Any]]  # Map "Speaker" tag to task parameters
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)
    silence_padding_ms: int = 800


class ScriptReadPipeline:
    """Pipeline for generating multi-speaker dialogue from a formatted script."""

    async def run(self, engine: Any, request: ScriptReadRequest) -> RunResult:
        """
        Executes the script reading pipeline.

        Args:
            engine: The TTSEngine instance.
            request: The ScriptReadRequest configuration.

        Returns:
            A RunResult object containing the stitched dialogue audio.
        """
        # 1. Parse script
        rows = self._parse_script(request.script_text)
        log.info("Parsed %d lines from script", len(rows))

        all_wavs = []
        sample_rate = 24000

        for i, row in enumerate(rows):
            speaker_config = request.speaker_map.get(row.speaker)
            if not speaker_config:
                log.warning("Unknown speaker tag: %s, skipping", row.speaker)
                continue

            task, task_req = PipelineHelper.prepare_task_from_config(
                speaker_config, row.text, request.model, request.gen
            )
            wav, sr = await PipelineHelper.run_task_and_load(engine, task, task_req)
            sample_rate = sr
            all_wavs.append(wav)

            if request.silence_padding_ms > 0 and i < len(rows) - 1:
                silence_len = int(sr * (request.silence_padding_ms / 1000.0))
                all_wavs.append(np.zeros(silence_len))

        # 2. Stitch
        combined = np.concatenate(all_wavs)

        # 3. Finalize
        meta_extra = {
            "lines": len(rows),
            "duration_sec": len(combined) / sample_rate,
        }
        return PipelineHelper.finalize_run(
            engine,
            run_name="script_read",
            combined_wav=combined,
            sample_rate=sample_rate,
            params=request.model_dump(),
            meta_extra=meta_extra,
        )

    def _parse_script(self, text: str) -> list[ScriptRow]:
        """Parses a text script into individual speaker rows."""
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            match = re.match(r"^([^:]+):\s*(.*)$", line)
            if match:
                rows.append(
                    ScriptRow(
                        speaker=match.group(1).strip(), text=match.group(2).strip()
                    )
                )
        return rows
