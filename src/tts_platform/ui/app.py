from __future__ import annotations

import gradio as gr

from ..config import AppConfig
from .pages.audio_tools import create_audio_tools_page
from .pages.common import UIState
from .pages.custom_voice import create_custom_voice_page
from .pages.design_then_clone import create_design_then_clone_page
from .pages.live import create_live_page
from .pages.outputs import create_outputs_page
from .pages.pipelines import create_pipelines_page
from .pages.tokenizer import create_tokenizer_page
from .pages.voice_clone import create_voice_clone_page
from .pages.voice_design import create_voice_design_page
from .pages.voices import create_voices_page


def create_ui(cfg: AppConfig) -> gr.Blocks:
    state = UIState(cfg)

    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        gr.Markdown(
            f"""
# Qwen3-TTS Studio
Mode: `{state.mode}`
API: `{state.api_url}`

Tip: start with **0.6B** models if VRAM is tight, then move to **1.7B**.
"""
        )

        # Page-specific refresh functions and their output components
        refresh_registry = []

        with gr.Tab("CustomVoice"):
            refresh_registry.append(create_custom_voice_page(state))

        with gr.Tab("VoiceDesign"):
            refresh_registry.append(create_voice_design_page(state))

        with gr.Tab("VoiceClone"):
            refresh_registry.append(create_voice_clone_page(state))

        with gr.Tab("Design → Clone"):
            refresh_registry.append(create_design_then_clone_page(state))

        with gr.Tab("Pipelines"):
            refresh_registry.append(create_pipelines_page(state))

        with gr.Tab("Tokenizer"):
            refresh_registry.append(create_tokenizer_page(state))

        with gr.Tab("Voices Library"):
            refresh_registry.append(create_voices_page(state))

        with gr.Tab("Outputs Browser"):
            refresh_registry.append(create_outputs_page(state))

        with gr.Tab("Live Mode"):
            refresh_registry.append(create_live_page(state))

        with gr.Tab("Audio Tools"):
            create_audio_tools_page(state)

        with gr.Tab("Diagnostics"):
            gr.Markdown("### Diagnostics")
            with gr.Row():
                btn_global = gr.Button("Global Refresh", variant="secondary")
                btn_models = gr.Button("Show Models JSON")
                btn_voices = gr.Button("Show Voices JSON")

            out_diag = gr.Textbox(lines=20, label="Output")

            def list_m():
                return state.safe_json(state.get_models())

            def list_v():
                return state.safe_json(state.get_voices())

            btn_models.click(fn=list_m, outputs=out_diag)
            btn_voices.click(fn=list_v, outputs=out_diag)

            for fn, outputs in refresh_registry:
                if fn and outputs:
                    btn_global.click(fn=fn, outputs=outputs)

        # Wire initial loads
        for fn, outputs in refresh_registry:
            if fn and outputs:
                demo.load(fn=fn, outputs=outputs)

    return demo
