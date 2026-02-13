from __future__ import annotations

import gradio as gr

from ..constants import CUSTOMVOICE_SPEAKERS, LANG_CHOICES
from .common import UIState


def create_pipelines_page(state: UIState):
    async def do_long_form(text: str, speaker: str, lang: str, model_label: str):
        if not text.strip():
            return "", None, "No text"
        _, m_map = state.model_choices()
        model = m_map.get(model_label)

        from ...pipelines.long_form import LongFormPipeline, LongFormRequest
        pipe = LongFormPipeline()
        req = LongFormRequest(text=text, speaker=speaker, language=lang, model=model)

        try:
            if state.mode == "local_engine":
                res = await pipe.run(state.engine, req)
                return res.run_id, str(res.audio_path), state.safe_json(res.meta)
            else:
                return "", None, "Pipeline only available in local_engine mode for now"
        except Exception as e:
            return "", None, str(e)

    async def do_npc_pack(csv_file: str, speaker_json: str, model_label: str):
        if not csv_file or not speaker_json.strip():
            return "", "Missing inputs"
        import json
        try:
            speaker_map = json.loads(speaker_json)
        except Exception as e:
            return "", f"Invalid Speaker Map JSON: {e}"

        _, m_map = state.model_choices()
        model = m_map.get(model_label)

        from ...pipelines.npc_pack import NPCPackPipeline, NPCPackRequest
        pipe = NPCPackPipeline()
        req = NPCPackRequest(csv_path=csv_file, speaker_map=speaker_map, model=model)

        try:
            if state.mode != "local_engine":
                return "", "Pipeline only available in local_engine mode for now"
            res = await pipe.run(state.engine, req)
            return res.run_id, state.safe_json(res.meta)
        except Exception as e:
            return "", str(e)

    async def do_audiobook(chapter_files: list[str], speaker: str, lang: str, model_label: str):
        if not chapter_files:
            return "", None, "No chapters"
        _, m_map = state.model_choices()
        model = m_map.get(model_label)

        from ...pipelines.audiobook import AudiobookPipeline, AudiobookRequest
        pipe = AudiobookPipeline()
        req = AudiobookRequest(chapter_paths=chapter_files, speaker=speaker, language=lang, model=model)

        try:
            if state.mode != "local_engine":
                return "", None, "Requires local_engine"
            res = await pipe.run(state.engine, req)
            return res.run_id, str(res.audio_path), state.safe_json(res.meta)
        except Exception as e:
            return "", None, str(e)

    async def do_script_read(script_text: str, speaker_json: str, model_label: str):
        if not script_text.strip():
            return "", None, "No script"
        import json
        try:
            speaker_map = json.loads(speaker_json)
        except Exception as e:
            return "", None, f"JSON Error: {e}"

        _, m_map = state.model_choices()
        model = m_map.get(model_label)

        from ...pipelines.script_read import ScriptReadPipeline, ScriptReadRequest
        pipe = ScriptReadPipeline()
        req = ScriptReadRequest(script_text=script_text, speaker_map=speaker_map, model=model)

        try:
            if state.mode != "local_engine":
                return "", None, "Requires local_engine"
            res = await pipe.run(state.engine, req)
            return res.run_id, str(res.audio_path), state.safe_json(res.meta)
        except Exception as e:
            return "", None, str(e)

    async def do_subtitles(srt_file: str, speaker: str, lang: str, model_label: str, preserve: bool):
        if not srt_file:
            return "", None, "No SRT"
        _, m_map = state.model_choices()
        model = m_map.get(model_label)

        from ...pipelines.subtitles import SubtitlesPipeline, SubtitlesRequest
        pipe = SubtitlesPipeline()
        req = SubtitlesRequest(srt_path=srt_file, speaker=speaker, language=lang, model=model, preserve_timing=preserve)

        try:
            if state.mode != "local_engine":
                return "", None, "Requires local_engine"
            res = await pipe.run(state.engine, req)
            return res.run_id, str(res.audio_path), state.safe_json(res.meta)
        except Exception as e:
            return "", None, str(e)

    with gr.Column():
        gr.Markdown("### Advanced Pipelines\nComplex multi-step workflows for content creation.")

        with gr.Tab("Long-form Narration"):
            with gr.Row():
                with gr.Column(scale=2):
                    lf_text = gr.Textbox(label="Long Text", lines=15)
                with gr.Column(scale=1):
                    lf_spk = gr.Dropdown(choices=CUSTOMVOICE_SPEAKERS, value="Ryan", label="Speaker")
                    lf_lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
                    lf_model = gr.Dropdown(label="Model")
                    lf_btn = gr.Button("Generate Narrator Audio", variant="primary")
            with gr.Row():
                lf_id = gr.Textbox(label="Run ID")
                lf_audio = gr.Audio(label="Audio")
            lf_meta = gr.Textbox(label="Metadata", lines=3)

        with gr.Tab("NPC Pack"):
             gr.Markdown("Generate batches of audio for NPCs from a CSV file.")
             with gr.Row():
                 npc_csv = gr.File(label="CSV File (character_id, text, line_id)", file_types=[".csv"])
                 npc_json = gr.Textbox(label="Speaker Map (JSON)", lines=5, placeholder='{"char_1": {"type": "custom_voice", "speaker": "Ryan"}}')
             npc_model = gr.Dropdown(label="Model")
             npc_btn = gr.Button("Process NPC Pack", variant="primary")
             npc_id = gr.Textbox(label="Run ID")
             npc_meta = gr.Textbox(label="Manifest Summary", lines=5)

        with gr.Tab("Audiobook / Web-novel"):
             gr.Markdown("Batch process multiple chapter files into a merged narration.")
             with gr.Row():
                 ab_files = gr.File(label="Chapter Files (.txt)", file_count="multiple")
                 with gr.Column():
                     ab_spk = gr.Dropdown(choices=CUSTOMVOICE_SPEAKERS, value="Ryan", label="Narrator Voice")
                     ab_lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
                     ab_model = gr.Dropdown(label="Model")
                     ab_btn = gr.Button("Generate Audiobook", variant="primary")
             with gr.Row():
                 ab_id = gr.Textbox(label="Run ID")
                 ab_audio = gr.Audio(label="Full Merged Audio")
             ab_meta = gr.Textbox(label="Batch Info", lines=3)

        with gr.Tab("Script Table-read"):
             gr.Markdown("Auto-detect speakers in a script and generate dialogue.")
             with gr.Row():
                 scr_text = gr.Textbox(label="Script Content", lines=10, placeholder="[Ryan] Hello.\n[Vivian] Hi there!")
                 scr_json = gr.Textbox(label="Speaker Config (JSON)", lines=5, placeholder='{"Ryan": {"type": "custom_voice", "speaker": "Ryan"}}')
             scr_model = gr.Dropdown(label="Model")
             scr_btn = gr.Button("Process Script", variant="primary")
             with gr.Row():
                 scr_id = gr.Textbox(label="Run ID")
                 scr_audio = gr.Audio(label="Merged Table Read")
             scr_meta = gr.Textbox(label="Result Info", lines=3)

        with gr.Tab("Subtitle-to-Speech"):
             gr.Markdown("Convert an SRT file into an audio track synchronized with the subtitles.")
             with gr.Row():
                 sub_file = gr.File(label="SRT File", file_types=[".srt"])
                 with gr.Column():
                     sub_spk = gr.Dropdown(choices=CUSTOMVOICE_SPEAKERS, value="Ryan", label="Voice")
                     sub_lang = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Language")
                     sub_preserve = gr.Checkbox(label="Preserve Original Timing (Padded Silence)", value=True)
             sub_model = gr.Dropdown(label="Model")
             sub_btn = gr.Button("Generate Dubbing Track", variant="primary")
             with gr.Row():
                 sub_id = gr.Textbox(label="Run ID")
                 sub_audio = gr.Audio(label="Timed Audio")
             sub_meta = gr.Textbox(label="Stats", lines=3)

        def refresh():
            labels, _ = state.model_choices()
            cv_labels = [label for label in labels if "customvoice" in label.lower() or "base" in label.lower()]
            return (
                gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None),
                gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None),
                gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None),
                gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None),
                gr.Dropdown(choices=cv_labels, value=cv_labels[0] if cv_labels else None),
            )

        lf_btn.click(fn=do_long_form, inputs=[lf_text, lf_spk, lf_lang, lf_model], outputs=[lf_id, lf_audio, lf_meta])
        npc_btn.click(fn=do_npc_pack, inputs=[npc_csv, npc_json, npc_model], outputs=[npc_id, npc_meta])
        ab_btn.click(fn=do_audiobook, inputs=[ab_files, ab_spk, ab_lang, ab_model], outputs=[ab_id, ab_audio, ab_meta])
        scr_btn.click(fn=do_script_read, inputs=[scr_text, scr_json, scr_model], outputs=[scr_id, scr_audio, scr_meta])
        sub_btn.click(fn=do_subtitles, inputs=[sub_file, sub_spk, sub_lang, sub_model, sub_preserve], outputs=[sub_id, sub_audio, sub_meta])

        return refresh
