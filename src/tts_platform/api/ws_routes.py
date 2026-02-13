from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ..engine.engine import TTSEngine

log = logging.getLogger("tts_platform.api.ws")


def get_engine() -> TTSEngine:
    raise RuntimeError("engine dependency not configured")


router = APIRouter()


async def _recv_start(ws: WebSocket) -> dict[str, Any]:
    msg = await ws.receive_text()
    data = json.loads(msg)
    if data.get("type") != "start":
        raise ValueError("first message must be type=start")
    return data


async def _stream_iterator(
    ws: WebSocket, gen: AsyncIterator[tuple[np.ndarray, int]], channels: int = 1
) -> None:
    import asyncio

    header_sent = False
    stop_requested = False

    # We need to run two things concurrently:
    # 1. Pulling chunks from the engine (gen)
    # 2. Listening for a "stop" message from the client

    async def recv_loop():
        nonlocal stop_requested
        try:
            async for msg in ws.iter_text():
                data = json.loads(msg)
                if data.get("type") == "stop":
                    log.info("Client requested stream stop")
                    stop_requested = True
                    break
        except Exception:
            pass

    recv_task = asyncio.create_task(recv_loop())

    try:
        async for wav, sr in gen:
            if stop_requested:
                break

            if not header_sent:
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
                header_sent = True

            pcm = TTSEngine.wav_to_pcm16_bytes(wav)
            if pcm:
                await ws.send_bytes(pcm)
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    await ws.send_text(json.dumps({"type": "end", "ok": not stop_requested, "stopped": stop_requested}))


@router.websocket("/ws/tts/custom_voice")
async def ws_custom_voice(
    ws: WebSocket, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        text = start.get("text", "")
        language = start.get("language", "Auto")
        speaker = start.get("speaker", "Ryan")
        instruct = start.get("instruct", "")
        model = start.get("model")

        gen = engine.stream_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            model=model,
            gen={},
        )
        await _stream_iterator(ws, gen)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()


@router.websocket("/ws/tts/voice_design")
async def ws_voice_design(
    ws: WebSocket, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        text = start.get("text", "")
        language = start.get("language", "Auto")
        instruct = start.get("instruct", "")
        model = start.get("model")

        gen = engine.stream_voice_design(
            text=text, language=language, instruct=instruct, model=model, gen={}
        )
        await _stream_iterator(ws, gen)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()


@router.websocket("/ws/tts/voice_clone")
async def ws_voice_clone(
    ws: WebSocket, engine: Annotated[TTSEngine, Depends(get_engine)]
) -> None:
    await ws.accept()
    try:
        start = await _recv_start(ws)
        text = start.get("text", "")
        language = start.get("language", "Auto")
        model = start.get("model")

        voice_profile = start.get("voice_profile")
        ref_audio = start.get("ref_audio")
        ref_text = start.get("ref_text")
        x_vector_only_mode = bool(start.get("x_vector_only_mode", False))
        use_cached_prompt = bool(start.get("use_cached_prompt", True))

        gen = engine.stream_voice_clone(
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
        await _stream_iterator(ws, gen)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "end", "ok": False, "error": str(e)}))
    finally:
        await ws.close()
