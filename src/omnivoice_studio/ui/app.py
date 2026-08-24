"""Main entry point for the Gradio UI application."""

from __future__ import annotations

from typing import Any

import gradio as gr

# Bypass Pylint false positives for Gradio dynamic attributes
ga: Any = gr

from ..config import AppConfig
from .pages.audio_tools import create_audio_tools_page
from .pages.common import UIState
from .pages.custom_voice import create_custom_voice_page
from .pages.design_then_clone import create_design_then_clone_page
from .pages.live import create_live_page
from .pages.outputs import create_outputs_page
from .pages.pipelines import create_pipelines_page
from .pages.singing import create_singing_page
from .pages.tokenizer import create_tokenizer_page
from .pages.voice_clone import create_voice_clone_page
from .pages.voice_conversion import create_voice_conversion_page
from .pages.voice_design import create_voice_design_page
from .pages.voice_sculptor import create_voice_sculptor_page
from .pages.voices import create_voices_page


def _create_diagnostics_tab(state: UIState, refresh_registry: list):
    """Creates the diagnostics tab and wires global refresh."""
    with ga.Tab("Diagnostics"):
        ga.Markdown("### Diagnostics")
        with ga.Row():
            btn_global = ga.Button("Global Refresh", variant="secondary")
            btn_models = ga.Button("Show Models JSON")
            btn_voices = ga.Button("Show Voices JSON")

        out_diag = ga.Textbox(lines=20, label="Output")

        getattr(btn_models, "click")(
            fn=lambda: state.safe_json(state.get_models()), outputs=out_diag
        )
        getattr(btn_voices, "click")(
            fn=lambda: state.safe_json(state.get_voices()), outputs=out_diag
        )

        for fn, outputs in refresh_registry:
            if fn and outputs:
                getattr(btn_global, "click")(fn=fn, outputs=outputs)


def create_ui(cfg: AppConfig) -> ga.Blocks:
    """
    Creates the main Gradio UI layout and wires page components.

    Args:
        cfg: The application configuration.

    Returns:
        The configured Gradio Blocks object.
    """
    state = UIState(cfg)
    refresh_registry = []

    with ga.Blocks(title="OmniVoice Studio 🎙️", theme=ga.themes.Soft()) as demo:
        ga.Markdown(
            f"""
# 🎙️ OmniVoice Studio
**Your All-in-One Professional Voice AI Platform**

Mode: `{state.mode}` | API: `{state.api_url}`
"""
        )

        with ga.Tabs():
            with ga.Tab("TTS & Cloning"):
                with ga.Tabs():
                    with ga.Tab("CustomVoice"):
                        refresh_registry.append(create_custom_voice_page(state))
                    with ga.Tab("VoiceDesign"):
                        refresh_registry.append(create_voice_design_page(state))
                    with ga.Tab("VoiceClone"):
                        refresh_registry.append(create_voice_clone_page(state))
                    with ga.Tab("Design → Clone"):
                        refresh_registry.append(create_design_then_clone_page(state))

            with ga.Tab("Specialized Tasks"):
                with ga.Tabs():
                    with ga.Tab("Singing Synthesis"):
                        refresh_registry.append(create_singing_page(state))
                    with ga.Tab("Voice Conversion"):
                        refresh_registry.append(create_voice_conversion_page(state))
                    with ga.Tab("Voice Sculptor"):
                        refresh_registry.append(create_voice_sculptor_page(state))

            with ga.Tab("Pipelines"):
                refresh_registry.append(create_pipelines_page(state))

            with ga.Tab("Streaming & Assets"):
                with ga.Tabs():
                    with ga.Tab("Live Mode"):
                        refresh_registry.append(create_live_page(state))
                    with ga.Tab("Voices Library"):
                        refresh_registry.append(create_voices_page(state))
                    with ga.Tab("Outputs Browser"):
                        refresh_registry.append(create_outputs_page(state))
                    with ga.Tab("Audio Tools"):
                        create_audio_tools_page(state)
                    with ga.Tab("Tokenizer"):
                        refresh_registry.append(create_tokenizer_page(state))

            _create_diagnostics_tab(state, refresh_registry)

        # Wire initial loads
        for fn, outputs in refresh_registry:
            if fn and outputs:
                getattr(demo, "load")(fn=fn, outputs=outputs)

    return demo
