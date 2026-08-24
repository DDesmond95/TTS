"""Task implementation for Voice Design followed by Voice Clone in one workflow."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..storage.outputs import RunResult
from .base import Task
from .voice_design import VoiceDesignRequest, VoiceDesignTask

log = logging.getLogger("omnivoice_studio.tasks.design_then_clone")


class DesignThenCloneRequest(BaseModel):
    """Request schema for design-then-clone workflow."""

    design_text: str
    design_language: str = "Auto"
    design_instruct: str = ""
    clone_text: str | list[str]
    clone_language: str | list[str] = "Auto"
    voicedesign_model: str | None = None
    base_model: str | None = None
    gen_design: dict[str, Any] = Field(default_factory=dict)
    gen_clone: dict[str, Any] = Field(default_factory=dict)


class DesignThenCloneTask(Task[DesignThenCloneRequest, RunResult]):
    """Executes a two-step generation: first design a voice, then clone it immediately."""

    def validate(self, request: DesignThenCloneRequest) -> DesignThenCloneRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: DesignThenCloneRequest) -> RunResult:
        """Execute the design-then-clone task."""
        # Step 1: Design
        design_task = VoiceDesignTask()
        design_req = VoiceDesignRequest(
            text=request.design_text,
            language=request.design_language,
            instruct=request.design_instruct,
            model=request.voicedesign_model,
            gen=request.gen_design,
        )
        design_res = await design_task.run(engine, design_req)

        # Step 2: Build clone prompt in-memory and then clone
        base_id_or_path = engine.resolve_model(request.base_model, expected_kind="base")

        async with engine.sem:
            base_obj = await asyncio.to_thread(
                engine.get_or_load, base_id_or_path, "base"
            )

            if not design_res.audio_path:
                raise RuntimeError("design step did not produce audio.wav")

            prompt_items = await asyncio.to_thread(
                base_obj.create_voice_clone_prompt,
                ref_audio=str(design_res.audio_path),
                ref_text=request.design_text,
                x_vector_only_mode=False,
            )

            wavs, sr = await asyncio.to_thread(
                base_obj.generate_voice_clone,
                text=request.clone_text,
                language=request.clone_language,
                voice_clone_prompt=prompt_items,
                **request.gen_clone,
            )

        run_id, run_dir = self._prepare_run(
            engine,
            "design_then_clone",
            {
                "task": "design_then_clone",
                "voicedesign_run_id": design_res.run_id,
                "voicedesign_model": request.voicedesign_model,
                "base_model": base_id_or_path,
                "design_text": request.design_text,
                "design_language": request.design_language,
                "design_instruct": request.design_instruct,
                "clone_text": request.clone_text,
                "clone_language": request.clone_language,
                "gen_design": request.gen_design,
                "gen_clone": request.gen_clone,
            },
        )

        return engine.outputs.complete_run(run_id, run_dir, wavs, sr)
