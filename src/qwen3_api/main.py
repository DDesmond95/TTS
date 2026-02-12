import io
import os
import shutil
import uuid
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .model_wrapper import ModelWrapper

app = FastAPI(title="Qwen3 TTS API")
model_manager = ModelWrapper()

# Configuration
VOICES_DIR = "d:/CodeAlpha/Projects/YTProjects/TTS/voices"
RAW_VOICES_DIR = "d:/CodeAlpha/Projects/YTProjects/TTS/voices/raw"
TEMP_DIR = "d:/CodeAlpha/Projects/YTProjects/TTS/outputs/temp"
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(RAW_VOICES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


class GenerateCustomRequest(BaseModel):
    text: str
    speaker: str
    language: str = "Auto"
    instruct: Optional[str] = None


class GenerateDesignRequest(BaseModel):
    text: str
    instruct: str
    language: str = "Auto"


class GenerateCloneRequest(BaseModel):
    text: str
    language: str = "Auto"
    ref_text: Optional[str] = None
    use_xvector: bool = False
    voice_name: Optional[str] = None  # If provided, use a saved voice


def numpy_to_wav_bytes(wav: np.ndarray, sr: int):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def audio_stream_generator(wav: np.ndarray, sr: int, chunk_size: int = 4096):
    # Convert to bytes first
    wav_bytes = numpy_to_wav_bytes(wav, sr)
    for i in range(0, len(wav_bytes), chunk_size):
        yield wav_bytes[i : i + chunk_size]


@app.post("/generate/custom")
async def generate_custom(req: GenerateCustomRequest, stream: bool = False):
    model = model_manager.get_model_for_method("custom")
    wavs, sr = model.generate_custom_voice(
        text=req.text, speaker=req.speaker, language=req.language, instruct=req.instruct
    )

    if stream:
        return StreamingResponse(
            audio_stream_generator(wavs[0], sr), media_type="audio/wav"
        )

    return StreamingResponse(
        io.BytesIO(numpy_to_wav_bytes(wavs[0], sr)), media_type="audio/wav"
    )


@app.post("/generate/design")
async def generate_design(req: GenerateDesignRequest, stream: bool = False):
    model = model_manager.get_model_for_method("design")
    wavs, sr = model.generate_voice_design(
        text=req.text, instruct=req.instruct, language=req.language
    )

    if stream:
        return StreamingResponse(
            audio_stream_generator(wavs[0], sr), media_type="audio/wav"
        )

    return StreamingResponse(
        io.BytesIO(numpy_to_wav_bytes(wavs[0], sr)), media_type="audio/wav"
    )


@app.post("/generate/clone")
async def generate_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_text: Optional[str] = Form(None),
    use_xvector: bool = Form(False),
    voice_name: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    stream: bool = Form(False),
):
    model = model_manager.get_model_for_method("clone")

    voice_prompt = None
    ref_audio_path = None

    if voice_name:
        # Check if it's a prompt file
        voice_path = os.path.join(VOICES_DIR, f"{voice_name}.pt")
        raw_path = os.path.join(RAW_VOICES_DIR, f"{voice_name}.wav")

        if os.path.exists(voice_path):
            payload = torch.load(voice_path, map_location="cpu", weights_only=True)
            from qwen_tts import VoiceClonePromptItem

            voice_prompt = []
            for d in payload["items"]:
                voice_prompt.append(
                    VoiceClonePromptItem(
                        ref_code=(
                            torch.tensor(d["ref_code"])
                            if d.get("ref_code") is not None
                            else None
                        ),
                        ref_spk_embedding=torch.tensor(d["ref_spk_embedding"]),
                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                        icl_mode=bool(d.get("icl_mode", True)),
                        ref_text=d.get("ref_text", None),
                    )
                )
        elif os.path.exists(raw_path):
            ref_audio_path = raw_path
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Voice {voice_name} not found in voices or raw_voices",
            )

    if not voice_prompt and (ref_audio or ref_audio_path):
        if ref_audio:
            temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(ref_audio.file, f)
            ref_audio_path = temp_path

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=use_xvector,
        )

        # Cleanup if it was a temp file
        if ref_audio and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
    elif voice_prompt:
        wavs, sr = model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=voice_prompt
        )
    else:
        raise HTTPException(
            status_code=400, detail="Either ref_audio or voice_name must be provided"
        )

    if stream:
        return StreamingResponse(
            audio_stream_generator(wavs[0], sr), media_type="audio/wav"
        )

    return StreamingResponse(
        io.BytesIO(numpy_to_wav_bytes(wavs[0], sr)), media_type="audio/wav"
    )


@app.post("/voices/save")
async def save_voice(
    name: str = Form(...),
    ref_text: Optional[str] = Form(None),
    use_xvector: bool = Form(False),
    ref_audio: UploadFile = File(...),
):
    model = model_manager.get_model_for_method("clone")

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(ref_audio.file, f)

    try:
        items = model.create_voice_clone_prompt(
            ref_audio=temp_path, ref_text=ref_text, x_vector_only_mode=use_xvector
        )

        from dataclasses import asdict

        payload = {
            "items": [asdict(it) for it in items],
        }

        # Convert tensors to lists for serializability if needed,
        # but torch.save handles tensors.
        # We'll use torch.save as it's standard for this model.
        save_path = os.path.join(VOICES_DIR, f"{name}.pt")
        torch.save(payload, save_path)

        return {
            "status": "success",
            "message": f"Voice saved as {name}",
            "path": save_path,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/voices/save_raw")
async def save_raw_voice(name: str = Form(...), ref_audio: UploadFile = File(...)):
    save_path = os.path.join(RAW_VOICES_DIR, f"{name}.wav")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(ref_audio.file, f)
    return {
        "status": "success",
        "message": f"Raw voice saved as {name}",
        "path": save_path,
    }


@app.get("/voices")
async def list_voices():
    prompts = [
        f.replace(".pt", "") for f in os.listdir(VOICES_DIR) if f.endswith(".pt")
    ]
    raws = [
        f.replace(".wav", "") for f in os.listdir(RAW_VOICES_DIR) if f.endswith(".wav")
    ]
    return {"voice_prompts": prompts, "raw_voices": raws}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
