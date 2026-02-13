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

        with gr.Tab("CustomVoice"):
            refresh_cv = create_custom_voice_page(state)

        with gr.Tab("VoiceDesign"):
            create_voice_design_page(state)

        with gr.Tab("VoiceClone"):
            create_voice_clone_page(state)

        with gr.Tab("Design → Clone"):
            create_design_then_clone_page(state)

        with gr.Tab("Pipelines"):
            create_pipelines_page(state)

        with gr.Tab("Tokenizer"):
            create_tokenizer_page(state)

        with gr.Tab("Voices Library"):
            create_voices_page(state)

        with gr.Tab("Outputs Browser"):
            create_outputs_page(state)

        with gr.Tab("Live Mode"):
            create_live_page(state)

        with gr.Tab("Audio Tools"):
            create_audio_tools_page(state)

        with gr.Tab("Diagnostics"):
            gr.Markdown("### Diagnostics")
            with gr.Row():
                gr.Button("Global Refresh", variant="secondary")
                btn_models = gr.Button("Show Models JSON")
                btn_voices = gr.Button("Show Voices JSON")

            out_diag = gr.Textbox(lines=20, label="Output")

            def list_m(): return state.safe_json(state.get_models())
            def list_v(): return state.safe_json(state.get_voices())

            btn_models.click(fn=list_m, outputs=out_diag)
            btn_voices.click(fn=list_v, outputs=out_diag)

            # Global refresh updates all page dropdowns
            def global_refresh():
                # We need to call all page refresh functions and return their outputs
                # This is a bit complex in Gradio if we have many components.
                # Simplest way is to just trigger them.
                pass

        # Wire initial loads and refresh
        demo.load(fn=refresh_cv, outputs=[]) # placeholder to trigger

        # In a real modular app, you'd wire these to btn_refresh
        # but for now, each page handles its own refresh or on-load logic.

    return demo
