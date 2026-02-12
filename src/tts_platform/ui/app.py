from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import requests

from ..config import AppConfig
from ..engine.engine import TTSEngine


def create_ui(cfg: AppConfig) -> gr.Blocks:
    """
    UI modes:
      - local_engine: calls TTSEngine directly (single process)
      - api: calls REST API (recommended for docker compose)
    """
    mode = cfg.ui.mode
    api_url = cfg.ui.api_url.rstrip("/")

    engine = None
    if mode == "local_engine":
        engine = TTSEngine(
            models_dir=Path(cfg.paths.models_dir),
            voices_dir=Path(cfg.paths.voices_dir),
            outputs_dir=Path(cfg.paths.outputs_dir),
            runtime=cfg.runtime,
        )

    def _api_post(path: str, payload: dict) -> dict:
        r = requests.post(f"{api_url}{path}", json=payload, timeout=1800)
        r.raise_for_status()
        return r.json()

    def list_models() -> str:
        if mode == "local_engine":
            assert engine is not None
            return json.dumps(engine.list_models(), indent=2, ensure_ascii=False)
        return json.dumps(_api_post("/models", {}), indent=2, ensure_ascii=False)

    def list_voices() -> str:
        if mode == "local_engine":
            assert engine is not None
            return json.dumps(engine.list_voices(), indent=2, ensure_ascii=False)
        r = requests.get(f"{api_url}/voices", timeout=60)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2, ensure_ascii=False)

    async def do_custom_voice(
        text: str, language: str, speaker: str, instruct: str, model: str
    ) -> tuple[str, Optional[str]]:
        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
            "model": model or None,
            "gen": {},
        }
        if mode == "local_engine":
            assert engine is not None
            res = await engine.run_custom_voice(**payload)
            audio = str(res.audio_path) if res.audio_path else None
            return res.run_id, audio
        res = _api_post("/tts/custom_voice", payload)
        audio_url = res.get("audio_url")
        audio_path = None
        if audio_url:
            # Let gradio play via URL
            audio_path = f"{api_url}{audio_url}"
        return res["run_id"], audio_path

    async def do_voice_design(
        text: str, language: str, instruct: str, model: str
    ) -> tuple[str, Optional[str]]:
        payload = {
            "text": text,
            "language": language,
            "instruct": instruct,
            "model": model or None,
            "gen": {},
        }
        if mode == "local_engine":
            assert engine is not None
            res = await engine.run_voice_design(**payload)
            audio = str(res.audio_path) if res.audio_path else None
            return res.run_id, audio
        res = _api_post("/tts/voice_design", payload)
        audio_url = res.get("audio_url")
        audio_path = f"{api_url}{audio_url}" if audio_url else None
        return res["run_id"], audio_path

    async def do_voice_clone(
        text: str, language: str, voice_profile: str, model: str
    ) -> tuple[str, Optional[str]]:
        payload = {
            "text": text,
            "language": language,
            "voice_profile": voice_profile or None,
            "model": model or None,
            "gen": {},
        }
        if mode == "local_engine":
            assert engine is not None
            res = await engine.run_voice_clone(**payload)
            audio = str(res.audio_path) if res.audio_path else None
            return res.run_id, audio
        res = _api_post("/tts/voice_clone", payload)
        audio_url = res.get("audio_url")
        audio_path = f"{api_url}{audio_url}" if audio_url else None
        return res["run_id"], audio_path

    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        gr.Markdown("# Qwen3-TTS Studio")

        with gr.Tab("Diagnostics"):
            btn_models = gr.Button("List Models")
            out_models = gr.Textbox(lines=16, label="Models")
            btn_models.click(fn=list_models, outputs=out_models)

            btn_voices = gr.Button("List Voices")
            out_voices = gr.Textbox(lines=16, label="Voice Profiles")
            btn_voices.click(fn=list_voices, outputs=out_voices)

        with gr.Tab("CustomVoice"):
            t = gr.Textbox(label="Text", lines=6)
            language = gr.Textbox(value="English", label="Language (or Auto)")
            speaker = gr.Textbox(value="Ryan", label="Speaker")
            instruct = gr.Textbox(value="", label="Instruct (style)")
            model = gr.Textbox(value="", label="Model (optional, local path or HF id)")
            btn = gr.Button("Generate")
            run_id = gr.Textbox(label="Run ID")
            audio = gr.Audio(label="Audio", type="filepath")
            btn.click(
                fn=do_custom_voice,
                inputs=[t, language, speaker, instruct, model],
                outputs=[run_id, audio],
            )

        with gr.Tab("VoiceDesign"):
            t = gr.Textbox(label="Text", lines=6)
            language = gr.Textbox(value="English", label="Language (or Auto)")
            instruct = gr.Textbox(
                value="Neutral voice.", label="Voice description / Instruct"
            )
            model = gr.Textbox(value="", label="Model (optional)")
            btn = gr.Button("Generate")
            run_id = gr.Textbox(label="Run ID")
            audio = gr.Audio(label="Audio", type="filepath")
            btn.click(
                fn=do_voice_design,
                inputs=[t, language, instruct, model],
                outputs=[run_id, audio],
            )

        with gr.Tab("VoiceClone"):
            t = gr.Textbox(label="Text", lines=6)
            language = gr.Textbox(value="English", label="Language (or Auto)")
            voice_profile = gr.Textbox(
                value="", label="Clone Voice Profile ID (recommended)"
            )
            model = gr.Textbox(value="", label="Base model (optional)")
            btn = gr.Button("Generate")
            run_id = gr.Textbox(label="Run ID")
            audio = gr.Audio(label="Audio", type="filepath")
            btn.click(
                fn=do_voice_clone,
                inputs=[t, language, voice_profile, model],
                outputs=[run_id, audio],
            )

    return demo
