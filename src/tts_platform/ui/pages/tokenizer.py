from __future__ import annotations

from typing import Any

import gradio as gr

from .common import UIState


def create_tokenizer_page(state: UIState):
    async def do_encode(audio_path: str, model_label: str) -> tuple[str, str]:
        if not audio_path:
            return "", "Please upload audio."

        _, model_map = state.model_choices()
        model = model_map.get(model_label)

        payload: dict[str, Any] = {"audio": audio_path, "model": model}

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                from ...tasks.tokenizer import TokenizerEncodeRequest, TokenizerEncodeTask
                task = TokenizerEncodeTask()
                req = TokenizerEncodeRequest(**payload)
                res = await task.run(state.engine, req)
                return res.run_id, state.safe_json(res.meta)

            api_res = state.api_post("/tokenizer/encode", payload)
            return str(api_res.get("run_id", "")), state.safe_json(api_res)
        except Exception as e:
            return "", f"ERROR: {e}"

    async def do_decode(codes_path: str, model_label: str) -> tuple[str, str | None, str]:
        if not codes_path:
            return "", None, "Please provide codes JSON path."

        _, model_map = state.model_choices()
        model = model_map.get(model_label)

        payload: dict[str, Any] = {"codes_json_path": codes_path, "model": model}

        try:
            if state.mode == "local_engine":
                assert state.engine is not None
                from ...tasks.tokenizer import TokenizerDecodeRequest, TokenizerDecodeTask
                task = TokenizerDecodeTask()
                req = TokenizerDecodeRequest(**payload)
                res = await task.run(state.engine, req)
                audio = str(res.audio_path) if res.audio_path else None
                return res.run_id, audio, state.safe_json(res.meta)

            api_res = state.api_post("/tokenizer/decode", payload)
            audio_url = api_res.get("audio_url")
            audio_path = f"{state.api_url}{audio_url}" if audio_url else None
            return str(api_res.get("run_id", "")), audio_path, state.safe_json(api_res)
        except Exception as e:
            return "", None, f"ERROR: {e}"

    with gr.Column():
        gr.Markdown("### 12Hz Speech Tokenizer\nEncode audio into discrete codes or decode codes back into audio.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Encode")
                e_audio = gr.Audio(label="Input Audio", type="filepath")
                e_model = gr.Dropdown(label="Tokenizer Model")
                e_btn = gr.Button("Encode Audio")
                e_run_id = gr.Textbox(label="Run ID")
                e_meta = gr.Textbox(label="Result (Codes Path)", lines=3)

            with gr.Column():
                gr.Markdown("#### Decode")
                d_codes = gr.Textbox(label="Codes JSON Path", placeholder="Path to a .json file containing codes")
                d_model = gr.Dropdown(label="Tokenizer Model")
                d_btn = gr.Button("Decode Codes")
                d_run_id = gr.Textbox(label="Run ID")
                d_audio = gr.Audio(label="Decoded Audio", type="filepath")
                d_meta = gr.Textbox(label="Metadata", lines=3)

        def refresh():
            labels, _ = state.model_choices()
            tok_labels = [label for label in labels if "tokenizer" in label.lower()]
            return (
                gr.Dropdown(choices=tok_labels, value=tok_labels[0] if tok_labels else None),
                gr.Dropdown(choices=tok_labels, value=tok_labels[0] if tok_labels else None),
                gr.Dropdown(choices=tok_labels, value=tok_labels[0] if tok_labels else None),
            )

        e_btn.click(fn=do_encode, inputs=[e_audio, e_model], outputs=[e_run_id, e_meta])
        d_btn.click(fn=do_decode, inputs=[d_codes, d_model], outputs=[d_run_id, d_audio, d_meta])

        gr.Markdown("---")
        gr.Markdown("### 📊 Compression Explorer")
        with gr.Row():
            c_audio = gr.Audio(label="Input Audio", type="filepath")
            c_model = gr.Dropdown(label="Model")
        c_btn = gr.Button("Compare Original vs. Reconstructed", variant="primary")

        with gr.Row():
            c_out_orig = gr.Audio(label="Original")
            c_out_recon = gr.Audio(label="Reconstructed")

        c_stats = gr.Markdown("Stats: codes/sec, bit rate, MSE (placeholder)")

        async def do_compare(audio_path: str, model_label: str):
            if not audio_path:
                return None, None, "No audio"

            # 1. Encode
            _, meta_str = await do_encode(audio_path, model_label)
            # Find the path from meta
            import json
            try:
                meta = json.loads(meta_str)
                # In current implementation, meta might be the full API response or just data
                # Let's assume there's a file saved.
                run_dir = meta.get("run_dir") or meta.get("meta", {}).get("run_dir")
                if not run_dir:
                     return audio_path, None, "Could not find reconstruction path"

                codes_path = f"{run_dir}/audio_0.json" # heuristic
                # 2. Decode
                _, recon_audio, _ = await do_decode(codes_path, model_label)

                return audio_path, recon_audio, f"Successfully reconstructed via {model_label}"
            except Exception:
                return audio_path, None, "Reconstruction failed"

        c_btn.click(fn=do_compare, inputs=[c_audio, c_model], outputs=[c_out_orig, c_out_recon, c_stats])

        return refresh
