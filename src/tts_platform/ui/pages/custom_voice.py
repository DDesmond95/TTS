from __future__ import annotations

from typing import Any

import gradio as gr

from ..constants import CUSTOMVOICE_SPEAKERS, LANG_CHOICES, STYLE_PRESETS
from .common import UIState


def create_custom_voice_page(state: UIState):
    async def do_run(
        text: str,
        language: str,
        speaker: str,
        preset: str,
        instruct: str,
        model_label: str,
        max_tokens: int,
        top_p: float,
        temp: float,
    ) -> tuple[str, str | None, str]:
        if not text.strip():
            return "", None, "Please provide text."

        if (not instruct.strip()) and preset in STYLE_PRESETS:
            instruct = STYLE_PRESETS[preset]

        _, model_map = state.model_choices()
        model_value = model_map.get(model_label)

        payload: dict[str, Any] = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
            "model": model_value,
            "gen": {"max_new_tokens": max_tokens, "top_p": top_p, "temperature": temp},
        }

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                from ...tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
                task = CustomVoiceTask()
                req = CustomVoiceRequest(**payload)
                res = await task.run(state.engine, req)
                audio = str(res.audio_path) if res.audio_path else None
                return res.run_id, audio, state.safe_json(res.meta)

            api_res = state.api_post("/tts/custom_voice", payload)
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
            spk = gr.Dropdown(choices=CUSTOMVOICE_SPEAKERS, value="Ryan", label="Speaker")
            model = gr.Dropdown(label="Model")

    with gr.Row():
        preset = gr.Dropdown(choices=list(STYLE_PRESETS.keys()), value="(none)", label="Style preset")
        inst = gr.Textbox(label="Instruct (overrides preset if set)", lines=2, value="")

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
        labels, _ = state.model_choices()
        cv_labels = [label for label in labels if "customvoice" in label.lower()]
        return gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None)

    btn.click(
        fn=do_run,
        inputs=[t, lang, spk, preset, inst, model, tokens, tp, tmp],
        outputs=[run_id, audio, meta]
    )

    return refresh
