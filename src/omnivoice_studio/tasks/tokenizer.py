"""Task implementations for audio encoding/decoding using Qwen3-TTS Tokenizer."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..storage.outputs import RunResult
from .base import Task

log = logging.getLogger("omnivoice_studio.tasks.tokenizer")


class TokenizerEncodeRequest(BaseModel):
    """Request schema for audio encoding (WAV -> Codes)."""

    audio: str
    model: str | None = None


class TokenizerDecodeRequest(BaseModel):
    """Request schema for audio decoding (Codes -> WAV)."""

    codes_json_path: str
    model: str | None = None


class TokenizerEncodeTask(Task[TokenizerEncodeRequest, RunResult]):
    """Encodes a WAV file into discrete tokenizer codes."""

    def validate(self, request: TokenizerEncodeRequest) -> TokenizerEncodeRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: TokenizerEncodeRequest) -> RunResult:
        """Execute the audio encoding task."""
        model_id_or_path = engine.resolve_model(
            request.model, expected_kind="tokenizer"
        )
        params = {
            "task": "tokenizer_encode",
            "model": model_id_or_path,
            "audio": request.audio,
        }
        run_id, run_dir = self._prepare_run(engine, "tokenizer_encode", params)

        async with engine.sem:
            tok = await asyncio.to_thread(
                engine.get_or_load, model_id_or_path, "tokenizer"
            )
            enc = await asyncio.to_thread(tok.encode, request.audio)

        out = (run_dir / "codes.json").resolve()
        out.write_text(json.dumps(enc, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = {"codes_path": out.name}
        engine.outputs.write_meta(run_dir, meta)

        return RunResult(
            run_id=run_id, run_dir=run_dir, audio_path=None, sample_rate=None, meta=meta
        )


class TokenizerDecodeTask(Task[TokenizerDecodeRequest, RunResult]):
    """Decodes discrete tokenizer codes back into a WAV file."""

    def validate(self, request: TokenizerDecodeRequest) -> TokenizerDecodeRequest:
        """Validate the incoming request parameters."""
        return request

    async def run(self, engine: Any, request: TokenizerDecodeRequest) -> RunResult:
        """Execute the audio decoding task."""
        model_id_or_path = engine.resolve_model(
            request.model, expected_kind="tokenizer"
        )
        params = {
            "task": "tokenizer_decode",
            "model": model_id_or_path,
            "codes_json_path": request.codes_json_path,
        }
        run_id, run_dir = self._prepare_run(engine, "tokenizer_decode", params)

        codes_path = Path(request.codes_json_path).resolve()
        enc = json.loads(codes_path.read_text(encoding="utf-8"))

        async with engine.sem:
            tok = await asyncio.to_thread(
                engine.get_or_load, model_id_or_path, "tokenizer"
            )
            wavs, sr = await asyncio.to_thread(tok.decode, enc)

        return engine.outputs.complete_run(run_id, run_dir, wavs, sr)
