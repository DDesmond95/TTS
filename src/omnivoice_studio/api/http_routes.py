"""HTTP REST API routes for OmniVoice Studio."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..engine.engine import TTSEngine
from ..voices.schema import VoiceProfile
from .schemas import (
    AudiobookRequest,
    CustomVoiceRequest,
    DesignThenCloneRequest,
    LongFormRequest,
    NPCPackRequest,
    RunResponse,
    ScriptReadRequest,
    SingingSynthesisRequest,
    SubtitlesRequest,
    TokenizerDecodeRequest,
    TokenizerEncodeRequest,
    VoiceCloneRequest,
    VoiceConversionRequest,
    VoiceDesignRequest,
    VoiceSculptRequest,
    WarmupRequest,
)

log = logging.getLogger("omnivoice_studio.api.http")


def get_engine() -> TTSEngine:
    """Provides the TTSEngine instance from the app state via dependency injection."""
    # injected in app.state in app.py; this is overridden by dependency override
    raise RuntimeError("engine dependency not configured")


router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Basic health check endpoint."""
    return {"ok": True}


@router.get("/ready")
def ready(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    """Readiness check endpoint that verifies model and voice directories."""
    return {
        "ok": True,
        "models_dir": str(engine.registry.models_dir),
        "voices_dir": str(engine.voices.voices_dir),
    }


@router.get("/models")
def list_models(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    """Lists all available models in the registry."""
    return {"models": engine.list_models()}


@router.post("/models/warmup")
async def warmup(
    req: WarmupRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> dict:
    """Pre-warms a model into memory."""
    await engine.warmup(req.model)
    return {"ok": True, "model": req.model}


@router.get("/voices")
def list_voices_route(engine: Annotated[TTSEngine, Depends(get_engine)]) -> dict:
    """Lists all available voice profiles."""
    return {"voices": engine.list_voices()}


@router.get("/voices/{voice_id}")
def get_voice_route(
    voice_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> VoiceProfile:
    """Retrieves a specific voice profile by its ID."""
    p = engine.get_voice(voice_id)
    if not p:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return p


@router.post("/voices/{voice_id}")
def save_voice_route(
    voice_id: str,
    profile: VoiceProfile,
    engine: Annotated[TTSEngine, Depends(get_engine)],
) -> dict:
    """Saves or updates a voice profile."""
    engine.save_voice(voice_id, profile.model_dump())
    return {"ok": True}


@router.delete("/voices/{voice_id}")
def delete_voice_route(
    voice_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> dict:
    """Deletes a specific voice profile."""
    ok = engine.delete_voice(voice_id)
    return {"ok": ok}


@router.get("/runs/{run_id}/audio")
def get_run_audio(
    run_id: str, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> FileResponse:
    """Serves the generated audio file for a given run."""
    run_dir = (engine.outputs.runs_dir / run_id).resolve()
    audio = (run_dir / "audio.wav").resolve()
    if not audio.exists():
        # Try audio_0.wav as fallback
        audio = (run_dir / "audio_0.wav").resolve()
        if not audio.exists():
            raise HTTPException(status_code=404, detail="Audio not found for run")
    return FileResponse(str(audio), media_type="audio/wav", filename="audio.wav")


@router.post("/tts/custom_voice", response_model=RunResponse)
async def tts_custom_voice(
    req: CustomVoiceRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    """Handles custom voice generation requests."""
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
    """Handles voice design requests."""
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
    """Handles voice cloning requests."""
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
    """Runs a voice design task followed by a cloning task."""
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
    """Encodes audio using a tokenizer model."""
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
    """Decodes codes using a tokenizer model."""
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


@router.post("/tts/voice_conversion", response_model=RunResponse)
async def tts_voice_conversion(
    req: VoiceConversionRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    """Runs a voice conversion task."""
    res = await engine.run_voice_conversion(
        source_audio=req.source_audio,
        target_speaker_audio=req.target_speaker_audio,
        target_speaker_id=req.target_speaker_id,
        model=req.model,
        steps=req.steps,
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


@router.post("/tts/singing_synthesis", response_model=RunResponse)
async def tts_singing_synthesis(
    req: SingingSynthesisRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    """Runs a singing synthesis task."""
    res = await engine.run_singing_synthesis(
        lyrics=req.lyrics,
        score=req.score,
        ref_audio=req.ref_audio,
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


@router.post("/tts/voice_sculpting", response_model=RunResponse)
async def tts_voice_sculpting(
    req: VoiceSculptRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    """Runs a voice sculpting task (zero-shot editing)."""
    res = await engine.run_voice_sculpting(
        instruction=req.instruction,
        ref_audio=req.ref_audio,
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


# --- Pipeline Endpoints ---


@router.post("/pipelines/long_form", response_model=RunResponse)
async def pipe_long_form(
    req: LongFormRequest, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> RunResponse:
    """Runs a long-form synthesis pipeline."""
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
    """Runs a batch NPC lines generation pipeline."""
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
    """Runs a script reading pipeline (multi-speaker dialogue)."""
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
    """Runs an audiobook generation pipeline."""
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
    """Runs a subtitle dubbing pipeline."""
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
