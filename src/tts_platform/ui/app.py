from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

from ..config import AppConfig
from ..engine.engine import TTSEngine


LANG_CHOICES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Qwen3-TTS CustomVoice premium timbres (as per the Qwen3-TTS readme)
CUSTOMVOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

# Simple instruct presets (you can extend this)
STYLE_PRESETS = {
    "(none)": "",
    "Warm / friendly": "Warm, friendly tone. Natural pacing.",
    "Calm / supportive": "Calm, supportive tone. Slightly slower pacing. Gentle prosody.",
    "Confident host": "Confident, energetic host tone. Clear articulation.",
    "Soft / intimate": "Soft, close-mic intimate tone. Lower volume. Slow pace.",
    "Romantic": "Romantic tone. Warm breathy delivery. Moderate pace.",
    "Storyteller": "Storytelling voice. Varied prosody and emphasis. Natural pauses.",
    "Serious / news": "Professional serious delivery. Neutral emotion. Steady pace.",
    "Excited": "Excited and upbeat. Faster pace. Higher energy.",
}


def create_ui(cfg: AppConfig) -> gr.Blocks:
    """
    UI modes:
      - local_engine: calls TTSEngine directly (single process)
      - api: calls REST API (recommended for docker compose)
    """
    mode = (cfg.ui.mode or "local_engine").strip()
    api_url = (cfg.ui.api_url or "http://localhost:8001").rstrip("/")

    engine: Optional[TTSEngine] = None
    if mode == "local_engine":
        engine = TTSEngine(
            models_dir=Path(cfg.paths.models_dir),
            voices_dir=Path(cfg.paths.voices_dir),
            outputs_dir=Path(cfg.paths.outputs_dir),
            runtime=cfg.runtime,
        )

    def _api_post(path: str, payload: dict) -> dict:
        r = requests.post(f"{api_url}{path}", json=payload, timeout=1800)
        r.raise_for_status()
        return r.json()

    def _api_get(path: str) -> dict:
        r = requests.get(f"{api_url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def _get_models() -> List[dict]:
        if mode == "local_engine":
            assert engine is not None
            return engine.list_models()
        return _api_get("/models").get("models", [])

    def _get_voices() -> List[dict]:
        if mode == "local_engine":
            assert engine is not None
            return engine.list_voices()
        return _api_get("/voices").get("voices", [])

    def _model_choices() -> Tuple[List[str], Dict[str, str]]:
        """
        Returns:
          - choices labels for dropdown
          - label->model_value mapping (we pass model_value to API/engine)
        """
        models = _get_models()
        mapping: Dict[str, str] = {}
        labels: List[str] = []
        for m in models:
            name = str(m.get("name", ""))
            kind = str(m.get("kind", "unknown"))
            path = str(m.get("path", name))
            label = f"{kind} | {name}"
            labels.append(label)
            mapping[label] = path
        labels.sort()
        return labels, mapping

    def _voice_choices(
        profile_type: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Returns:
          - choices labels for dropdown
          - label->voice_id mapping
        """
        voices = _get_voices()
        mapping: Dict[str, str] = {}
        labels: List[str] = []

        for v in voices:
            vid = str(v.get("id", ""))
            vtype = str(v.get("type", ""))
            display = str(v.get("display_name", vid))
            if profile_type and vtype != profile_type:
                continue
            label = f"{vtype} | {display} ({vid})"
            labels.append(label)
            mapping[label] = vid

        labels.sort()
        # allow "none"
        return ["(none)"] + labels, {"(none)": ""} | mapping

    def _safe_json(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return str(obj)

    # -------- Diagnostics handlers --------

    def diag_list_models() -> str:
        try:
            return _safe_json(_get_models())
        except Exception as e:
            return f"ERROR: {e}"

    def diag_list_voices() -> str:
        try:
            return _safe_json(_get_voices())
        except Exception as e:
            return f"ERROR: {e}"

    def refresh_models() -> Tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown]:
        labels, _ = _model_choices()
        # one dropdown reused across tabs; return updates for each instance
        return (
            gr.Dropdown(choices=labels, value=labels[0] if labels else None),
            gr.Dropdown(choices=labels, value=labels[0] if labels else None),
            gr.Dropdown(choices=labels, value=labels[0] if labels else None),
        )

    def refresh_voices() -> Tuple[gr.Dropdown]:
        labels, _ = _voice_choices(profile_type="clone")
        return (gr.Dropdown(choices=labels, value="(none)"),)

    # -------- Generation helpers --------

    async def do_custom_voice(
        text: str,
        language: str,
        speaker: str,
        preset: str,
        instruct: str,
        model_label: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Tuple[str, Optional[str], str]:
        if not text.strip():
            return "", None, "Please provide text."

        # apply preset if instruct is empty (or if user explicitly wants preset)
        if (not instruct.strip()) and preset in STYLE_PRESETS:
            instruct = STYLE_PRESETS[preset]

        labels, model_map = _model_choices()
        model_value = model_map.get(model_label, None)

        gen = {
            "max_new_tokens": int(max_new_tokens),
            "top_p": float(top_p),
            "temperature": float(temperature),
        }

        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
            "model": model_value,
            "gen": gen,
        }

        try:
            if mode == "local_engine":
                assert engine is not None
                res = await engine.run_custom_voice(**payload)
                audio = str(res.audio_path) if res.audio_path else None
                meta = _safe_json(
                    {
                        "run_id": res.run_id,
                        "meta": res.meta,
                        "run_dir": str(res.run_dir),
                    }
                )
                return res.run_id, audio, meta

            res = _api_post("/tts/custom_voice", payload)
            audio_url = res.get("audio_url")
            audio_path = f"{api_url}{audio_url}" if audio_url else None
            meta = _safe_json(res)
            return str(res.get("run_id", "")), audio_path, meta
        except Exception as e:
            return "", None, f"ERROR: {e}"

    async def do_voice_design(
        text: str,
        language: str,
        preset: str,
        instruct: str,
        model_label: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Tuple[str, Optional[str], str]:
        if not text.strip():
            return "", None, "Please provide text."

        if (not instruct.strip()) and preset in STYLE_PRESETS:
            instruct = STYLE_PRESETS[preset]

        labels, model_map = _model_choices()
        model_value = model_map.get(model_label, None)

        gen = {
            "max_new_tokens": int(max_new_tokens),
            "top_p": float(top_p),
            "temperature": float(temperature),
        }

        payload = {
            "text": text,
            "language": language,
            "instruct": instruct,
            "model": model_value,
            "gen": gen,
        }

        try:
            if mode == "local_engine":
                assert engine is not None
                res = await engine.run_voice_design(**payload)
                audio = str(res.audio_path) if res.audio_path else None
                meta = _safe_json(
                    {
                        "run_id": res.run_id,
                        "meta": res.meta,
                        "run_dir": str(res.run_dir),
                    }
                )
                return res.run_id, audio, meta

            res = _api_post("/tts/voice_design", payload)
            audio_url = res.get("audio_url")
            audio_path = f"{api_url}{audio_url}" if audio_url else None
            meta = _safe_json(res)
            return str(res.get("run_id", "")), audio_path, meta
        except Exception as e:
            return "", None, f"ERROR: {e}"

    async def do_voice_clone(
        text: str,
        language: str,
        voice_label: str,
        model_label: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Tuple[str, Optional[str], str]:
        if not text.strip():
            return "", None, "Please provide text."

        _, model_map = _model_choices()
        model_value = model_map.get(model_label, None)

        _, voice_map = _voice_choices(profile_type="clone")
        voice_id = voice_map.get(voice_label, "")

        gen = {
            "max_new_tokens": int(max_new_tokens),
            "top_p": float(top_p),
            "temperature": float(temperature),
        }

        payload = {
            "text": text,
            "language": language,
            "voice_profile": voice_id or None,
            "model": model_value,
            "gen": gen,
        }

        try:
            if mode == "local_engine":
                assert engine is not None
                res = await engine.run_voice_clone(**payload)
                audio = str(res.audio_path) if res.audio_path else None
                meta = _safe_json(
                    {
                        "run_id": res.run_id,
                        "meta": res.meta,
                        "run_dir": str(res.run_dir),
                    }
                )
                return res.run_id, audio, meta

            res = _api_post("/tts/voice_clone", payload)
            audio_url = res.get("audio_url")
            audio_path = f"{api_url}{audio_url}" if audio_url else None
            meta = _safe_json(res)
            return str(res.get("run_id", "")), audio_path, meta
        except Exception as e:
            return "", None, f"ERROR: {e}"

    # -------- UI construction --------

    # Prime initial dropdown data (best effort).
    try:
        model_labels_init, _model_map_init = _model_choices()
    except Exception:
        model_labels_init, _model_map_init = [], {}

    try:
        clone_voice_labels_init, _clone_voice_map_init = _voice_choices(
            profile_type="clone"
        )
    except Exception:
        clone_voice_labels_init, _clone_voice_map_init = ["(none)"], {"(none)": ""}

    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        gr.Markdown(
            f"""
# Qwen3-TTS Studio
Mode: `{mode}`
API: `{api_url}`

Tip: start with **0.6B** models if VRAM is tight, then move to **1.7B**.
"""
        )

        with gr.Tab("Diagnostics"):
            with gr.Row():
                btn_refresh_models = gr.Button("Refresh Models", variant="secondary")
                btn_refresh_voices = gr.Button(
                    "Refresh Voice Profiles", variant="secondary"
                )

            with gr.Row():
                btn_models = gr.Button("Show Models JSON")
                out_models = gr.Textbox(lines=14, label="Models")
            btn_models.click(fn=diag_list_models, outputs=out_models)

            with gr.Row():
                btn_voices = gr.Button("Show Voices JSON")
                out_voices = gr.Textbox(lines=14, label="Voice Profiles")
            btn_voices.click(fn=diag_list_voices, outputs=out_voices)

        with gr.Tab("CustomVoice"):
            with gr.Row():
                with gr.Column(scale=2):
                    t_cv = gr.Textbox(
                        label="Text", lines=7, placeholder="Enter text..."
                    )
                with gr.Column(scale=1):
                    lang_cv = gr.Dropdown(
                        choices=LANG_CHOICES, value="English", label="Language"
                    )
                    speaker_cv = gr.Dropdown(
                        choices=CUSTOMVOICE_SPEAKERS, value="Ryan", label="Speaker"
                    )
                    model_cv = gr.Dropdown(
                        choices=model_labels_init,
                        value=(model_labels_init[0] if model_labels_init else None),
                        label="Model",
                    )

            with gr.Row():
                preset_cv = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="(none)",
                    label="Style preset",
                )
                instruct_cv = gr.Textbox(
                    label="Instruct (overrides preset if set)", lines=2, value=""
                )

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    max_tokens_cv = gr.Slider(
                        64, 2048, value=1024, step=16, label="max_new_tokens"
                    )
                    top_p_cv = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p")
                    temp_cv = gr.Slider(
                        0.05, 1.5, value=0.8, step=0.01, label="temperature"
                    )

            btn_cv = gr.Button("Generate", variant="primary")
            with gr.Row():
                run_id_cv = gr.Textbox(label="Run ID")
                audio_cv = gr.Audio(label="Audio", type="filepath")
            meta_cv = gr.Textbox(lines=8, label="Response / Metadata")

            btn_cv.click(
                fn=do_custom_voice,
                inputs=[
                    t_cv,
                    lang_cv,
                    speaker_cv,
                    preset_cv,
                    instruct_cv,
                    model_cv,
                    max_tokens_cv,
                    top_p_cv,
                    temp_cv,
                ],
                outputs=[run_id_cv, audio_cv, meta_cv],
            )

        with gr.Tab("VoiceDesign"):
            with gr.Row():
                with gr.Column(scale=2):
                    t_vd = gr.Textbox(
                        label="Text", lines=7, placeholder="Enter text..."
                    )
                with gr.Column(scale=1):
                    lang_vd = gr.Dropdown(
                        choices=LANG_CHOICES, value="English", label="Language"
                    )
                    model_vd = gr.Dropdown(
                        choices=model_labels_init,
                        value=(model_labels_init[0] if model_labels_init else None),
                        label="Model",
                    )

            with gr.Row():
                preset_vd = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="(none)",
                    label="Style preset",
                )
                instruct_vd = gr.Textbox(
                    label="Voice description / Instruct", lines=3, value=""
                )

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    max_tokens_vd = gr.Slider(
                        64, 2048, value=1024, step=16, label="max_new_tokens"
                    )
                    top_p_vd = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p")
                    temp_vd = gr.Slider(
                        0.05, 1.5, value=0.8, step=0.01, label="temperature"
                    )

            btn_vd = gr.Button("Generate", variant="primary")
            with gr.Row():
                run_id_vd = gr.Textbox(label="Run ID")
                audio_vd = gr.Audio(label="Audio", type="filepath")
            meta_vd = gr.Textbox(lines=8, label="Response / Metadata")

            btn_vd.click(
                fn=do_voice_design,
                inputs=[
                    t_vd,
                    lang_vd,
                    preset_vd,
                    instruct_vd,
                    model_vd,
                    max_tokens_vd,
                    top_p_vd,
                    temp_vd,
                ],
                outputs=[run_id_vd, audio_vd, meta_vd],
            )

        with gr.Tab("VoiceClone"):
            with gr.Row():
                with gr.Column(scale=2):
                    t_vc = gr.Textbox(
                        label="Text", lines=7, placeholder="Enter text..."
                    )
                with gr.Column(scale=1):
                    lang_vc = gr.Dropdown(
                        choices=LANG_CHOICES, value="English", label="Language"
                    )
                    voice_vc = gr.Dropdown(
                        choices=clone_voice_labels_init,
                        value="(none)",
                        label="Clone voice profile",
                    )
                    model_vc = gr.Dropdown(
                        choices=model_labels_init,
                        value=(model_labels_init[0] if model_labels_init else None),
                        label="Base model",
                    )

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    max_tokens_vc = gr.Slider(
                        64, 2048, value=1024, step=16, label="max_new_tokens"
                    )
                    top_p_vc = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p")
                    temp_vc = gr.Slider(
                        0.05, 1.5, value=0.8, step=0.01, label="temperature"
                    )

            btn_vc = gr.Button("Generate", variant="primary")
            with gr.Row():
                run_id_vc = gr.Textbox(label="Run ID")
                audio_vc = gr.Audio(label="Audio", type="filepath")
            meta_vc = gr.Textbox(lines=8, label="Response / Metadata")

            btn_vc.click(
                fn=do_voice_clone,
                inputs=[
                    t_vc,
                    lang_vc,
                    voice_vc,
                    model_vc,
                    max_tokens_vc,
                    top_p_vc,
                    temp_vc,
                ],
                outputs=[run_id_vc, audio_vc, meta_vc],
            )

        # Wire refresh buttons to update dropdowns in relevant tabs
        # (We update 3 model dropdowns + clone voice dropdown.)
        btn_refresh_models.click(
            fn=refresh_models, outputs=[model_cv, model_vd, model_vc]
        )
        btn_refresh_voices.click(fn=refresh_voices, outputs=[voice_vc])

    return demo
