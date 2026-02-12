from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..engine.engine import TTSEngine
from .schemas import (
    CustomVoiceRequest,
    DesignThenCloneRequest,
    RunResponse,
    TokenizerDecodeRequest,
    TokenizerEncodeRequest,
    VoiceCloneRequest,
    VoiceDesignRequest,
    WarmupRequest,
)

log = logging.getLogger("tts_platform.api.http")


def get_engine() -> TTSEngine:
    # injected in app.state in app.py; this is overridden by dependency override
    raise RuntimeError("engine dependency not configured")


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/ready")
def ready(engine: TTSEngine = Depends(get_engine)) -> dict:
    # minimal readiness: dirs exist
    return {
        "ok": True,
        "models_dir": str(engine.registry.models_dir),
        "voices_dir": str(engine.voices.voices_dir),
    }


@router.get("/models")
def list_models(engine: TTSEngine = Depends(get_engine)) -> dict:
    return {"models": engine.list_models()}


@router.post("/models/warmup")
async def warmup(req: WarmupRequest, engine: TTSEngine = Depends(get_engine)) -> dict:
    return await engine.warmup(req.model)


@router.get("/voices")
def list_voices(engine: TTSEngine = Depends(get_engine)) -> dict:
    return {"voices": engine.list_voices()}


@router.get("/runs/{run_id}/audio")
def get_run_audio(run_id: str, engine: TTSEngine = Depends(get_engine)) -> FileResponse:
    run_dir = (engine.outputs.runs_dir / run_id).resolve()
    audio = (run_dir / "audio.wav").resolve()
    if not audio.exists():
        raise HTTPException(status_code=404, detail="audio.wav not found for run")
    return FileResponse(str(audio), media_type="audio/wav", filename="audio.wav")


@router.post("/tts/custom_voice", response_model=RunResponse)
async def tts_custom_voice(
    req: CustomVoiceRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    if isinstance(req.text, str) and len(req.text) > engine.runtime.text_max_chars:
        raise HTTPException(status_code=400, detail="text too long")
    res = await engine.run_custom_voice(
        text=req.text,
        language=req.language,
        speaker=req.speaker,
        instruct=req.instruct,
        model=req.model,
        gen=req.gen,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/tts/voice_design", response_model=RunResponse)
async def tts_voice_design(
    req: VoiceDesignRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    res = await engine.run_voice_design(
        text=req.text,
        language=req.language,
        instruct=req.instruct,
        model=req.model,
        gen=req.gen,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/tts/voice_clone", response_model=RunResponse)
async def tts_voice_clone(
    req: VoiceCloneRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    res = await engine.run_voice_clone(
        text=req.text,
        language=req.language,
        ref_audio=req.ref_audio,
        ref_text=req.ref_text,
        voice_profile=req.voice_profile,
        model=req.model,
        x_vector_only_mode=req.x_vector_only_mode,
        use_cached_prompt=req.use_cached_prompt,
        gen=req.gen,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/tts/design_then_clone", response_model=RunResponse)
async def tts_design_then_clone(
    req: DesignThenCloneRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    res = await engine.run_design_then_clone(
        design_text=req.design_text,
        design_language=req.design_language,
        design_instruct=req.design_instruct,
        clone_text=req.clone_text,
        clone_language=req.clone_language,
        voicedesign_model=req.voicedesign_model,
        base_model=req.base_model,
        gen_design=req.gen_design,
        gen_clone=req.gen_clone,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/tokenizer/encode", response_model=RunResponse)
async def tok_encode(
    req: TokenizerEncodeRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    res = await engine.tokenizer_encode(audio=req.audio, model=req.model)
    return RunResponse(
        run_id=res.run_id,
        sample_rate=None,
        audio_url=None,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/tokenizer/decode", response_model=RunResponse)
async def tok_decode(
    req: TokenizerDecodeRequest, engine: TTSEngine = Depends(get_engine)
) -> RunResponse:
    res = await engine.tokenizer_decode(
        codes_json_path=req.codes_json_path, model=req.model
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )
