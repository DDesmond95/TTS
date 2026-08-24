from __future__ import annotations

from typing import Any

import gradio as gr

from .common import UIState


def create_voice_sculptor_page(state: UIState):
    async def do_run(
        instruction: str,
        ref_audio: str | None,
        model_label: str,
    ):
        if not instruction.strip() or not ref_audio:
            return None, "Please provide both instruction and reference audio."

        _, model_map = state.model_choices()
        model_value = model_map.get(model_label)

        payload: dict[str, Any] = {
            "instruction": instruction,
            "ref_audio": ref_audio,
            "model": model_value,
        }

        from ...tasks.voice_sculptor import VoiceSculptRequest, VoiceSculptTask

        _, audio, meta = await state.run_task(
            "/tts/voice_sculpting", VoiceSculptTask, VoiceSculptRequest, payload
        )
        return audio, meta

    gr.Markdown("### 🛠️ VoiceSculptor - Voice Editing")
    with gr.Row():
        with gr.Column():
            ref = gr.Audio(label="Reference Audio", type="filepath")
        with gr.Column():
            inst = gr.Textbox(
                label="Sculpting Instruction",
                placeholder="e.g. 'Make the voice deeper and more whispery'",
                lines=5,
            )

    with gr.Row():
        model = gr.Dropdown(label="Sculpting Model")

    btn = gr.Button("Sculpt Voice", variant="primary")
    audio = gr.Audio(label="Sculpted Audio")
    meta = gr.Textbox(label="Metadata", lines=5)

    async def refresh():
        labels, _ = state.model_choices()
        vs_labels = [
            label
            for label in labels
            if "voicesculptor" in label.lower() or "llasa" in label.lower()
        ]
        return gr.update(choices=vs_labels, value=vs_labels[0] if vs_labels else None)

    btn.click(fn=do_run, inputs=[inst, ref, model], outputs=[audio, meta])
    return refresh, [model]
