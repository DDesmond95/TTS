from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..engine.engine import TTSEngine
from .schemas import (
    AudiobookRequest,
    CustomVoiceRequest,
    DesignThenCloneRequest,
    LongFormRequest,
    NPCPackRequest,
    RunResponse,
    ScriptReadRequest,
    SubtitlesRequest,
    TokenizerDecodeRequest,
    TokenizerEncodeRequest,
    VoiceCloneRequest,
    VoiceDesignRequest,
    VoiceProfile,
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
def ready(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    # minimal readiness: dirs exist
    return {
        "ok": True,
        "models_dir": str(engine.registry.models_dir),
        "voices_dir": str(engine.voices.voices_dir),
    }


@router.get("/models")
def list_models(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    return {"models": engine.list_models()}


@router.post("/models/warmup")
async def warmup(req: WarmupRequest, engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    return await engine.warmup(req.model)


@router.get("/voices")
def list_voices(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    return {"voices": engine.list_voices()}


@router.get("/voices/{voice_id}")
def get_voice(voice_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]) -> VoiceProfile:
    p = engine.voices.get(voice_id)
    if not p:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return p


@router.post("/voices/{voice_id}")
def save_voice(
    voice_id: str, profile: VoiceProfile, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> dict:
    engine.save_voice(voice_id, profile.model_dump())
    return {"ok": True}


@router.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    ok = engine.delete_voice(voice_id)
    return {"ok": ok}


@router.get("/voices/{voice_id}/export")
def export_voice(voice_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]) -> FileResponse:
    zip_path = engine.export_voice(voice_id)
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{voice_id}.zip"
    )


@router.post("/voices/import")
async def import_voice(
    file_path: str, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> dict:
    # In a real API this would be an UploadFile, but here we assume path
    vid = engine.import_voice(file_path)
    return {"ok": True, "voice_id": vid}


@router.get("/runs/{run_id}/export")
def export_run(run_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]) -> FileResponse:
    zip_path = engine.export_run(run_id)
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{run_id}.zip"
    )


@router.get("/runs/{run_id}/audio")
def get_run_audio(run_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]) -> FileResponse:
    run_dir = (engine.outputs.runs_dir / run_id).resolve()
    audio = (run_dir / "audio.wav").resolve()
    if not audio.exists():
        raise HTTPException(status_code=404, detail="audio.wav not found for run")
    return FileResponse(str(audio), media_type="audio/wav", filename="audio.wav")


@router.post("/tts/custom_voice", response_model=RunResponse)
async def tts_custom_voice(
    req: CustomVoiceRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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
    req: VoiceDesignRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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
    req: VoiceCloneRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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
    req: DesignThenCloneRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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
    req: TokenizerEncodeRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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
    req: TokenizerDecodeRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
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


# --- Pipeline Endpoints ---

@router.post("/pipelines/long_form", response_model=RunResponse)
async def pipe_long_form(
    req: LongFormRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    res = await engine.run_long_form(
        text=req.text,
        task_type=req.task_type,
        speaker=req.speaker,
        language=req.language,
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


@router.post("/pipelines/npc_pack", response_model=RunResponse)
async def pipe_npc_pack(
    req: NPCPackRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    res = await engine.run_npc_pack(
        csv_path=req.csv_path,
        speaker_map=req.speaker_map,
        model=req.model,
        gen=req.gen,
    )
    return RunResponse(
        run_id=res.run_id,
        sample_rate=None,  # batch output
        audio_url=None,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/pipelines/script_read", response_model=RunResponse)
async def pipe_script_read(
    req: ScriptReadRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    res = await engine.run_script_read(
        script_text=req.script_text,
        speaker_map=req.speaker_map,
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


@router.post("/pipelines/audiobook", response_model=RunResponse)
async def pipe_audiobook(
    req: AudiobookRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    res = await engine.run_audiobook(
        chapter_paths=req.chapter_paths,
        task_type=req.task_type,
        speaker=req.speaker,
        language=req.language,
        model=req.model,
        gen=req.gen,
        merge_all=req.merge_all,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )


@router.post("/pipelines/subtitles", response_model=RunResponse)
async def pipe_subtitles(
    req: SubtitlesRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    res = await engine.run_subtitles(
        srt_path=req.srt_path,
        speaker=req.speaker,
        language=req.language,
        model=req.model,
        gen=req.gen,
        preserve_timing=req.preserve_timing,
    )
    audio_url = f"/runs/{res.run_id}/audio" if res.audio_path else None
    return RunResponse(
        run_id=res.run_id,
        sample_rate=res.sample_rate,
        audio_url=audio_url,
        run_dir=str(res.run_dir),
        meta=res.meta,
    )
