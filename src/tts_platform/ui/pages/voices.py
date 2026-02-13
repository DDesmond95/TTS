from __future__ import annotations

from datetime import datetime

import gradio as gr

from ...voices.schema import CloneConfig, DesignTemplateConfig, VoiceDefaults, VoiceMeta, VoiceProfile
from .common import UIState


def create_voices_page(state: UIState):
    gr.Markdown("## Voice Library")

    with gr.Row():
        with gr.Column(scale=1):
            voice_list = gr.Dropdown(label="Select Voice Profile", choices=[v["id"] for v in state.get_voices()])
            refresh_btn = gr.Button("Refresh List")
            export_btn = gr.Button("Export Profile (.zip)")
            delete_btn = gr.Button("Delete Profile", variant="stop")

        with gr.Column(scale=2):
            v_id = gr.Textbox(label="Profile ID (slug)")
            v_name = gr.Textbox(label="Display Name")
            v_type = gr.Dropdown(label="Type", choices=["customvoice", "clone", "design_template"])

            with gr.Group(visible=True):
                gr.Markdown("### Defaults")
                v_lang = gr.Dropdown(label="Language", choices=["Auto", "English", "Chinese", "Japanese", "Korean"], value="Auto")
                v_speaker = gr.Textbox(label="Speaker ID (for customvoice)")
                v_instruct = gr.Textbox(label="Instruction Text")

            with gr.Group(visible=False) as group_clone:
                gr.Markdown("### Clone Settings")
                v_ref_audio = gr.Textbox(label="Ref Audio Path (relative to voices/ )")
                v_ref_text = gr.Textbox(label="Ref Text Path (optional)")
                v_xvector = gr.Checkbox(label="X-Vector Only Mode")

            with gr.Group(visible=False) as group_design:
                gr.Markdown("### Design Template Settings")
                v_template = gr.Textbox(label="Instruction Template", lines=3)
                v_example = gr.Textbox(label="Example Text", lines=2)

            save_btn = gr.Button("Save Profile", variant="primary")
            status = gr.Markdown("")

    def on_type_change(t):
        return {
            group_clone: gr.update(visible=(t == "clone")),
            group_design: gr.update(visible=(t == "design_template"))
        }

    v_type.change(fn=on_type_change, inputs=v_type, outputs=[group_clone, group_design])

    def load_profile(vid):
        if not vid:
            return [gr.update()] * 11

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                p = state.engine.voices.get(vid)
                if not p:
                    return [gr.update()] * 11
                p_dict = p.model_dump()
            else:
                p_dict = state.api_get(f"/voices/{vid}")

            df = p_dict.get("defaults", {})
            cl = p_dict.get("clone", {}) or {}
            ds = p_dict.get("design_template", {}) or {}

            return [
                p_dict.get("id"),
                p_dict.get("display_name"),
                p_dict.get("type"),
                df.get("language", "Auto"),
                df.get("speaker", ""),
                df.get("instruct", ""),
                cl.get("ref_audio_path", ""),
                cl.get("ref_text_path", ""),
                cl.get("x_vector_only_mode", False),
                ds.get("instruct_template", ""),
                ds.get("example_text", ""),
            ]
        except Exception:
            return [gr.update()] * 11

    voice_list.change(
        fn=load_profile,
        inputs=voice_list,
        outputs=[v_id, v_name, v_type, v_lang, v_speaker, v_instruct, v_ref_audio, v_ref_text, v_xvector, v_template, v_example]
    )

    def export_profile(vid):
        if not vid:
            return None
        if state.mode == "local_engine":
            assert state.engine is not None
            return str(state.engine.export_voice(vid))
        else:
            return f"{state.api_url}/voices/{vid}/export"

    file_out = gr.File(label="Exported Pack", visible=False)
    export_btn.click(fn=export_profile, inputs=voice_list, outputs=file_out).then(
        fn=lambda: gr.update(visible=True), outputs=file_out
    )

    def save_profile(vid, name, vtype, lang, speaker, instruct, ref_audio, ref_text, xvector, template, example):
        now = datetime.now().isoformat()
        p = VoiceProfile(
            id=vid,
            type=vtype,
            display_name=name,
            defaults=VoiceDefaults(language=lang, speaker=speaker, instruct=instruct),
            clone=CloneConfig(ref_audio_path=ref_audio, ref_text_path=ref_text, x_vector_only_mode=xvector) if vtype == "clone" else None,
            design_template=DesignTemplateConfig(instruct_template=template, example_text=example) if vtype == "design_template" else None,
            meta=VoiceMeta(created_at=now, updated_at=now)
        )
        res = state.api_post(f"/voices/{vid}", p.model_dump())
        if res.get("ok"):
            return "Saved successfully!", gr.update(choices=[v["id"] for v in state.get_voices()])
        return f"Error: {res.get('detail', 'Unknown error')}", gr.update()

    save_btn.click(
        fn=save_profile,
        inputs=[v_id, v_name, v_type, v_lang, v_speaker, v_instruct, v_ref_audio, v_ref_text, v_xvector, v_template, v_example],
        outputs=[status, voice_list]
    )

    def delete_profile(vid):
        if not vid:
            return "Select a profile first", gr.update()
        _ = state.api_delete(f"/voices/{vid}")
        return "Deleted", gr.update(choices=[v["id"] for v in state.get_voices()])

    delete_btn.click(fn=delete_profile, inputs=voice_list, outputs=[status, voice_list])

    def refresh():
        return gr.update(choices=[v["id"] for v in state.get_voices()])

    refresh_btn.click(fn=refresh, outputs=voice_list)
