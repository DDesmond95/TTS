from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .base import Task

log = logging.getLogger("tts_platform.tasks.design_then_clone")


class DesignThenCloneRequest(BaseModel):
    design_text: str
    design_language: str = "Auto"
    design_instruct: str = ""
    clone_text: str | list[str]
    clone_language: str | list[str] = "Auto"
    voicedesign_model: str | None = None
    base_model: str | None = None
    gen_design: dict[str, Any] = Field(default_factory=dict)
    gen_clone: dict[str, Any] = Field(default_factory=dict)


class DesignThenCloneTask(Task[DesignThenCloneRequest, Any]):
    def validate(self, request: DesignThenCloneRequest) -> DesignThenCloneRequest:
        return request

    async def run(self, engine: Any, request: DesignThenCloneRequest) -> Any:
        # Step 1: Design
        from .voice_design import VoiceDesignRequest, VoiceDesignTask
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
        import asyncio


        base_id_or_path = engine._resolve_model(request.base_model, expected_kind="base")

        async with engine._sem:
            base_obj = await asyncio.to_thread(
                engine._get_or_load, base_id_or_path, "base"
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

        run_id, run_dir = engine.outputs.new_run_dir("design_then_clone")
        engine.outputs.write_params(
            run_dir,
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

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )
