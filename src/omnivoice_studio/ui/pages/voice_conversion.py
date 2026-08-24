from __future__ import annotations

from typing import Any

import gradio as gr

from .common import UIState


def create_voice_conversion_page(state: UIState):
    async def do_run(
        source_audio: str | None,
        target_audio: str | None,
        steps: int,
        model_label: str,
    ):
        if not source_audio:
            return None, "Please provide source audio."

        _, model_map = state.model_choices()
        model_value = model_map.get(model_label)

        payload: dict[str, Any] = {
            "source_audio": source_audio,
            "target_speaker_audio": target_audio,
            "steps": steps,
            "model": model_value,
        }

        from ...tasks.meanvc import VoiceConversionRequest, VoiceConversionTask

        _, audio, meta = await state.run_task(
            "/tts/voice_conversion",
            VoiceConversionTask,
            VoiceConversionRequest,
            payload,
        )
        return audio, meta

    gr.Markdown("### 🔊 MeanVC Voice Conversion")
    with gr.Row():
        with gr.Column():
            src = gr.Audio(
                label="Source Audio (The voice you want to change)", type="filepath"
            )
        with gr.Column():
            tgt = gr.Audio(
                label="Target Audio (The identity you want to copy)", type="filepath"
            )

    with gr.Row():
        model = gr.Dropdown(label="MeanVC Model")
        steps = gr.Slider(1, 50, value=5, step=1, label="DDIM Steps")

    btn = gr.Button("Convert Voice", variant="primary")
    audio = gr.Audio(label="Output Audio")
    meta = gr.Textbox(label="Metadata", lines=5)

    async def refresh():
        labels, _ = state.model_choices()
        mv_labels = [label for label in labels if "meanvc" in label.lower()]
        return gr.update(choices=mv_labels, value=mv_labels[0] if mv_labels else None)

    btn.click(fn=do_run, inputs=[src, tgt, steps, model], outputs=[audio, meta])
    return refresh, [model]
