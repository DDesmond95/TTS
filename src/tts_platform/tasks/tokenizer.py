from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .base import Task

log = logging.getLogger("tts_platform.tasks.tokenizer")


class TokenizerEncodeRequest(BaseModel):
    audio: str
    model: str | None = None


class TokenizerDecodeRequest(BaseModel):
    codes_json_path: str
    model: str | None = None


class TokenizerEncodeTask(Task[TokenizerEncodeRequest, Any]):
    def validate(self, request: TokenizerEncodeRequest) -> TokenizerEncodeRequest:
        return request

    async def run(self, engine: Any, request: TokenizerEncodeRequest) -> Any:
        model_id_or_path = engine._resolve_model(request.model, expected_kind="tokenizer")
        run_id, run_dir = engine.outputs.new_run_dir("tokenizer_encode")
        params = {"task": "tokenizer_encode", "model": model_id_or_path, "audio": request.audio}
        engine.outputs.write_params(run_dir, params)

        import asyncio
        async with engine._sem:
            tok = await asyncio.to_thread(
                engine._get_or_load, model_id_or_path, "tokenizer"
            )
            enc = await asyncio.to_thread(tok.encode, request.audio)

        out = (run_dir / "codes.json").resolve()
        out.write_text(json.dumps(enc, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = {"codes_path": out.name}
        engine.outputs.write_meta(run_dir, meta)

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id, run_dir=run_dir, audio_path=None, sample_rate=None, meta=meta
        )


class TokenizerDecodeTask(Task[TokenizerDecodeRequest, Any]):
    def validate(self, request: TokenizerDecodeRequest) -> TokenizerDecodeRequest:
        return request

    async def run(self, engine: Any, request: TokenizerDecodeRequest) -> Any:
        model_id_or_path = engine._resolve_model(request.model, expected_kind="tokenizer")
        run_id, run_dir = engine.outputs.new_run_dir("tokenizer_decode")
        params = {
            "task": "tokenizer_decode",
            "model": model_id_or_path,
            "codes_json_path": request.codes_json_path,
        }
        engine.outputs.write_params(run_dir, params)

        codes_path = Path(request.codes_json_path).resolve()
        enc = json.loads(codes_path.read_text(encoding="utf-8"))

        import asyncio
        async with engine._sem:
            tok = await asyncio.to_thread(
                engine._get_or_load, model_id_or_path, "tokenizer"
            )
            wavs, sr = await asyncio.to_thread(tok.decode, enc)

        audio_path = engine.outputs.save_wav(
            run_dir, np.asarray(wavs[0]), int(sr), filename="audio.wav"
        )
        meta = {"sample_rate": int(sr), "files": [audio_path.name]}
        engine.outputs.write_meta(run_dir, meta)

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )
