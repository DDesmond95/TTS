from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ..engine.engine import TTSEngine

log = logging.getLogger("tts_platform.api.ws")


def get_engine() -> TTSEngine:
    raise RuntimeError("engine dependency not configured")


router = APIRouter()


async def _recv_start(ws: WebSocket) -> Dict[str, Any]:
    msg = await ws.receive_text()
    data = json.loads(msg)
    if data.get("type") != "start":
        raise ValueError("first message must be type=start")
    return data


async def _stream_pcm16(
    ws: WebSocket, sr: int, wav: np.ndarray, chunk_ms: int, channels: int = 1
) -> None:
    await ws.send_text(
        json.dumps(
            {
                "type": "header",
                "format": "pcm16",
                "sample_rate": sr,
                "channels": channels,
            }
        )
    )

    pcm = TTSEngine.wav_to_pcm16_bytes(wav)
    bytes_per_sample = 2 * channels
    samples_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    for i in range(0, len(pcm), bytes_per_chunk):
        await ws.send_bytes(pcm[i : i + bytes_per_chunk])

    await ws.send_text(json.dumps({"type": "end", "ok": True}))


@router.websocket("/ws/tts/custom_voice")
async def ws_custom_voice(
    ws: WebSocket, engine: TTSEngine = Depends(get_engine)
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        chunk_ms = int(start.get("chunk_ms", 60))
        text = start.get("text", "")
        language = start.get("language", "Auto")
        speaker = start.get("speaker", "Ryan")
        instruct = start.get("instruct", "")
        model = start.get("model")

        # NOTE: This is "pseudo-streaming": generate first, then stream chunks.
        res = await engine.run_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            model=model,
            gen={},
        )
        audio_path = res.audio_path or (res.run_dir / "audio.wav")
        import soundfile as sf

        wav, sr = sf.read(str(audio_path), always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        await _stream_pcm16(
            ws,
            int(sr),
            np.asarray(wav, dtype=np.float32),
            chunk_ms=chunk_ms,
            channels=1,
        )
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()


@router.websocket("/ws/tts/voice_design")
async def ws_voice_design(
    ws: WebSocket, engine: TTSEngine = Depends(get_engine)
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        chunk_ms = int(start.get("chunk_ms", 60))
        text = start.get("text", "")
        language = start.get("language", "Auto")
        instruct = start.get("instruct", "")
        model = start.get("model")

        res = await engine.run_voice_design(
            text=text, language=language, instruct=instruct, model=model, gen={}
        )
        audio_path = res.audio_path or (res.run_dir / "audio.wav")
        import soundfile as sf

        wav, sr = sf.read(str(audio_path), always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        await _stream_pcm16(
            ws,
            int(sr),
            np.asarray(wav, dtype=np.float32),
            chunk_ms=chunk_ms,
            channels=1,
        )
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()


@router.websocket("/ws/tts/voice_clone")
async def ws_voice_clone(
    ws: WebSocket, engine: TTSEngine = Depends(get_engine)
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        chunk_ms = int(start.get("chunk_ms", 60))
        text = start.get("text", "")
        language = start.get("language", "Auto")
        model = start.get("model")

        voice_profile = start.get("voice_profile")
        ref_audio = start.get("ref_audio")
        ref_text = start.get("ref_text")
        x_vector_only_mode = bool(start.get("x_vector_only_mode", False))
        use_cached_prompt = bool(start.get("use_cached_prompt", True))

        res = await engine.run_voice_clone(
            text=text,
            language=language,
            voice_profile=voice_profile,
            ref_audio=ref_audio,
            ref_text=ref_text,
            model=model,
            x_vector_only_mode=x_vector_only_mode,
            use_cached_prompt=use_cached_prompt,
            gen={},
        )
        audio_path = res.audio_path or (res.run_dir / "audio.wav")
        import soundfile as sf

        wav, sr = sf.read(str(audio_path), always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        await _stream_pcm16(
            ws,
            int(sr),
            np.asarray(wav, dtype=np.float32),
            chunk_ms=chunk_ms,
            channels=1,
        )
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()
