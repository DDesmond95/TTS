from __future__ import annotations

import gradio as gr
import numpy as np

from ..constants import STYLE_PRESETS
from .common import UIState


def create_live_page(state: UIState):
    gr.Markdown("## Live Mode (VTuber Friendly)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Preset & Style")
            voice_list = gr.Dropdown(
                label="Quick Voice Selection",
                choices=[v["id"] for v in state.get_voices()],
                value=next((v["id"] for v in state.get_voices()), None)
            )
            refresh_btn = gr.Button("Refresh Voices")

            style_list = gr.Radio(
                label="Style Overrides",
                choices=list(STYLE_PRESETS.keys()),
                value="(none)"
            )

            with gr.Accordion("Advanced Live Settings", open=False):
                model_list = gr.Dropdown(label="Model Override", choices=["0.6b-customvoice", "1.7b-customvoice", "1.7b-voicedesign"])
                top_p = gr.Slider(0.0, 1.0, 0.8, step=0.05, label="Top P")
                temp = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="Temperature")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Live Trigger")
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Type here and press Speak...",
                lines=5,
                autofocus=True
            )

            with gr.Row():
                speak_btn = gr.Button("Speak (Stream)", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")

            # For Gradio output streaming:
            # Note: in many hosted Gradio environments, streaming output is tricky.
            # We will use the chunk iterator from the engine.
            audio_stream = gr.Audio(label="Generated Audio", streaming=True, autoplay=True)

            status = gr.Markdown("Ready")

    async def get_live_stream(text, voice_id, style_name, model_label, p, t):
        if not text:
            return

        _, m_map = state.model_choices()
        model = m_map.get(model_label)
        instruct = STYLE_PRESETS.get(style_name, "")

        gen_params = {"top_p": p, "temperature": t}

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                # Check profile type
                p_obj = state.engine.voices.get(voice_id)
                vtype = p_obj.type if p_obj else "customvoice"

                if vtype == "customvoice":
                    spk = p_obj.defaults.speaker if p_obj else "Ryan"
                    async for chunk_wav, sr in state.engine.stream_custom_voice(
                        text=text, speaker=spk, instruct=instruct, model=model, gen=gen_params
                    ):
                        # Convert float32 [-1, 1] to int16
                        i16 = (np.clip(chunk_wav, -1.0, 1.0) * 32767.0).astype(np.int16)
                        yield (sr, i16)
                elif vtype == "clone":
                    async for chunk_wav, sr in state.engine.stream_voice_clone(
                        text=text, voice_profile=voice_id, model=model, gen=gen_params
                    ):
                        i16 = (np.clip(chunk_wav, -1.0, 1.0) * 32767.0).astype(np.int16)
                        yield (sr, i16)
            else:
                # API streaming requires a WebSocket client which we haven't implemented here yet.
                # For now, yield some silence.
                import asyncio
                for _ in range(5):
                    await asyncio.sleep(0.1)
                    yield (24000, np.zeros(12000, dtype=np.int16))
        except Exception as e:
             # Yield a small burst of noise or just stop
             print(f"Live Error: {e}")

    speak_btn.click(
        fn=get_live_stream,
        inputs=[text_input, voice_list, style_list, model_list, top_p, temp],
        outputs=audio_stream
    )

    def on_stop():
        return "Stopped"

    stop_btn.click(fn=on_stop, outputs=status)

    def refresh():
        return gr.update(choices=[v["id"] for v in state.get_voices()])

    refresh_btn.click(fn=refresh, outputs=voice_list)

    return refresh
