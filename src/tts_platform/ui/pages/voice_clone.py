from __future__ import annotations

from typing import Any

import gradio as gr

from ..constants import LANG_CHOICES
from .common import UIState


def create_voice_clone_page(state: UIState):
    async def do_run(
        text: str,
        language: str,
        voice_label: str,
        model_label: str,
        max_tokens: int,
        top_p: float,
        temp: float,
    ) -> tuple[str, str | None, str]:
        if not text.strip():
            return "", None, "Please provide text."

        _, model_map = state.model_choices()
        model_value = model_map.get(model_label)

        _, voice_map = state.voice_choices(profile_type="clone")
        voice_id = voice_map.get(voice_label)

        payload: dict[str, Any] = {
            "text": text,
            "language": language,
            "voice_profile": voice_id or None,
            "model": model_value,
            "gen": {"max_new_tokens": max_tokens, "top_p": top_p, "temperature": temp},
        }

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                from ...tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask
                task = VoiceCloneTask()
                req = VoiceCloneRequest(**payload)
                res = await task.run(state.engine, req)
                audio = str(res.audio_path) if res.audio_path else None
                return res.run_id, audio, state.safe_json(res.meta)

            api_res = state.api_post("/tts/voice_clone", payload)
            audio_url = api_res.get("audio_url")
            audio_path = f"{state.api_url}{audio_url}" if audio_url else None
            return str(api_res.get("run_id", "")), audio_path, state.safe_json(api_res)
        except Exception as e:
            return "", None, f"ERROR: {e}"

    with gr.Row():
        with gr.Column(scale=2):
            t = gr.Textbox(label="Text", lines=7, placeholder="Enter text...")
        with gr.Column(scale=1):
            lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
            voice = gr.Dropdown(label="Clone voice profile")
            model = gr.Dropdown(label="Base model")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            tokens = gr.Slider(64, 2048, value=1024, step=16, label="max_new_tokens")
            tp = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p")
            tmp = gr.Slider(0.05, 1.5, value=0.8, step=0.01, label="temperature")

    btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        run_id = gr.Textbox(label="Run ID")
        audio = gr.Audio(label="Audio", type="filepath")
    meta = gr.Textbox(lines=8, label="Response / Metadata")

    def refresh():
        m_labels, _ = state.model_choices()
        base_labels = [label for label in m_labels if "base" in label.lower()]

        v_labels, _ = state.voice_choices(profile_type="clone")

        return (
            gr.Dropdown(choices=v_labels, value="(none)"),
            gr.Dropdown(choices=base_labels, value=base_labels[0] if base_labels else None)
        )

    btn.click(
        fn=do_run,
        inputs=[t, lang, voice, model, tokens, tp, tmp],
        outputs=[run_id, audio, meta]
    )

    return refresh
