from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .common import UIState


def create_outputs_page(state: UIState):
    def list_runs() -> list[tuple[str, str, str]]:
        if state.mode == "local_engine":
            runs_dir = Path(state.cfg.paths.outputs_dir) / "runs"
        else:
            # Fallback to local if API doesn't expose list_runs,
            # but ideally API should have /runs endpoint.
            runs_dir = Path(state.cfg.paths.outputs_dir) / "runs"

        if not runs_dir.exists():
            return []

        runs = []
        for d in sorted(runs_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if d.is_dir():
                # Extract task from folder name: timestamp_task_id
                parts = d.name.split("_")
                task = parts[1] if len(parts) > 1 else "unknown"
                runs.append((d.name, task, str(os.path.getmtime(d))))
        return runs

    def format_runs(runs):
        if not runs:
            return "No runs found."
        html = "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr><th>Run ID</th><th>Task</th><th>Action</th></tr>"
        for rid, task, _ in runs:
            html += f"<tr><td>{rid}</td><td>{task}</td><td>Select</td></tr>"
        html += "</table>"
        return html

    def load_run(run_id: str):
        runs_dir = Path(state.cfg.paths.outputs_dir) / "runs"
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            return None, "Run not found", ""

        audio_path = run_dir / "audio.wav"
        if not audio_path.exists():
            # Check for audio_0.wav
            audio_path = run_dir / "audio_0.wav"

        audio = str(audio_path) if audio_path.exists() else None

        params_path = run_dir / "params.json"
        params = params_path.read_text(encoding="utf-8") if params_path.exists() else "No params found"

        meta_path = run_dir / "meta.json"
        meta = meta_path.read_text(encoding="utf-8") if meta_path.exists() else "No meta found"

        return audio, params, meta

    with gr.Column():
        gr.Markdown("### Outputs Browser\nView and play previous generations.")

        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh List")
                run_list = gr.Dropdown(label="Select Run")

            with gr.Column(scale=2):
                audio_out = gr.Audio(label="Playback")
                with gr.Row():
                    params_out = gr.Textbox(label="Parameters", lines=10)
                    meta_out = gr.Textbox(label="Metadata", lines=10)

        def refresh_dropdown():
            runs = list_runs()
            choices = [r[0] for r in runs]
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        refresh_btn.click(fn=refresh_dropdown, outputs=run_list)
        run_list.change(fn=load_run, inputs=run_list, outputs=[audio_out, params_out, meta_out])

        return refresh_dropdown
