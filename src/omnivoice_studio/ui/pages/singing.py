from __future__ import annotations

import gradio as gr

from .common import UIState


def create_singing_page(state: UIState):
    async def do_run(
        lyrics: str,
        ref_audio: str | None,
        model_label: str,
    ):
        if not lyrics.strip():
            return None, "Please provide lyrics."

        _, model_map = state.model_choices()
        model_value = model_map.get(model_label)

        payload = {
            "lyrics": lyrics,
            "ref_audio": ref_audio,
            "model": model_value,
        }

        from ...tasks.tcsinger import SingingSynthesisRequest, SingingSynthesisTask

        _, audio, meta = await state.run_task(
            "/tts/singing_synthesis",
            SingingSynthesisTask,
            SingingSynthesisRequest,
            payload,
        )
        return audio, meta

    gr.Markdown("### 🎤 TCSinger2 Singing Synthesis")
    with gr.Row():
        with gr.Column(scale=2):
            lyrics_input = gr.Textbox(
                label="Lyrics", lines=7, placeholder="Enter lyrics here..."
            )
        with gr.Column(scale=1):
            ref = gr.Audio(label="Reference Singer Audio (Optional)", type="filepath")
            model = gr.Dropdown(label="TCSinger Model")

    btn = gr.Button("Generate Singing", variant="primary")
    audio = gr.Audio(label="Output Audio")
    meta = gr.Textbox(label="Metadata", lines=5)

    async def refresh():
        labels, _ = state.model_choices()
        tc_labels = [label for label in labels if "tcsinger" in label.lower()]
        return gr.update(choices=tc_labels, value=tc_labels[0] if tc_labels else None)

    btn.click(fn=do_run, inputs=[lyrics_input, ref, model], outputs=[audio, meta])
    return refresh, [model]
