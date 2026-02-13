from __future__ import annotations

from typing import Any

import gradio as gr

from ..constants import LANG_CHOICES
from .common import UIState


def create_design_then_clone_page(state: UIState):
    async def do_run(
        design_text: str,
        design_lang: str,
        design_instruct: str,
        clone_text: str,
        clone_lang: str,
        vd_model_label: str,
        base_model_label: str,
        max_tokens: int,
    ) -> tuple[str, str | None, str]:
        if not design_text.strip() or not clone_text.strip():
            return "", None, "Please provide both design and clone text."

        _, model_map = state.model_choices()
        vd_model = model_map.get(vd_model_label)
        base_model = model_map.get(base_model_label)

        payload: dict[str, Any] = {
            "design_text": design_text,
            "design_language": design_lang,
            "design_instruct": design_instruct,
            "clone_text": clone_text,
            "clone_language": clone_lang,
            "voicedesign_model": vd_model,
            "base_model": base_model,
            "gen_design": {"max_new_tokens": max_tokens},
            "gen_clone": {"max_new_tokens": max_tokens},
        }

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                from ...tasks.design_then_clone import DesignThenCloneRequest, DesignThenCloneTask
                task = DesignThenCloneTask()
                req = DesignThenCloneRequest(**payload)
                res = await task.run(state.engine, req)
                audio = str(res.audio_path) if res.audio_path else None
                return res.run_id, audio, state.safe_json(res.meta)

            api_res = state.api_post("/tts/design_then_clone", payload)
            audio_url = api_res.get("audio_url")
            audio_path = f"{state.api_url}{audio_url}" if audio_url else None
            return str(api_res.get("run_id", "")), audio_path, state.safe_json(api_res)
        except Exception as e:
            return "", None, f"ERROR: {e}"

    with gr.Column():
        gr.Markdown("### Design → Clone Workflow\nDesign a unique voice first, then use it as a reference to generate multiple lines.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Step 1: Design Reference")
                d_text = gr.Textbox(label="Design Text (e.g., 'Hello, this is my new voice.')", lines=3)
                d_lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
                d_inst = gr.Textbox(label="Voice Description", placeholder="e.g., 'A confident female voice with a British accent.'")
                vd_model = gr.Dropdown(label="VoiceDesign Model")

            with gr.Column():
                gr.Markdown("#### Step 2: Clone Target")
                c_text = gr.Textbox(label="Target Text (the content you actually want)", lines=5)
                c_lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
                base_model = gr.Dropdown(label="Base Model")

        with gr.Accordion("Settings", open=False):
            max_tokens = gr.Slider(64, 2048, value=1024, step=64, label="Max New Tokens")

        btn = gr.Button("Execute Workflow", variant="primary")

        with gr.Row():
            run_id = gr.Textbox(label="Run ID")
            audio = gr.Audio(label="Final Audio", type="filepath")
        meta = gr.Textbox(label="Metadata", lines=5)

        # Initialization helpers
        def refresh():
            labels, _ = state.model_choices()
            vd_labels = [label for label in labels if "voicedesign" in label.lower()]
            base_labels = [label for label in labels if "base" in label.lower()]
            return (
                gr.Dropdown(choices=vd_labels, value=vd_labels[0] if vd_labels else None),
                gr.Dropdown(choices=base_labels, value=base_labels[0] if base_labels else None),
            )

        btn.click(
            fn=do_run,
            inputs=[d_text, d_lang, d_inst, c_text, c_lang, vd_model, base_model, max_tokens],
            outputs=[run_id, audio, meta]
        )

        return refresh
