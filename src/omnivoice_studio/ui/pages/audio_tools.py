from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from .common import UIState


def create_audio_tools_page(state: UIState):
    gr.Markdown("## 🛠️ Audio Utilities")

    with gr.Tab("Trim Silence"):
        gr.Markdown("Remove leading and trailing silence from an audio file.")
        with gr.Row():
            t_in = gr.Audio(label="Input Audio", type="filepath")
            with gr.Column():
                t_thr = gr.Slider(0.001, 0.1, 0.01, step=0.001, label="Threshold")
                t_pad = gr.Slider(0, 500, 50, step=10, label="Padding (ms)")
                t_btn = gr.Button("Trim Silence", variant="primary")
        t_out = gr.Audio(label="Trimmed Output")

    with gr.Tab("Normalize"):
        gr.Markdown("Normalize audio loudness to a target peak level.")
        with gr.Row():
            n_in = gr.Audio(label="Input Audio", type="filepath")
            with gr.Column():
                n_peak = gr.Slider(-20.0, 0.0, -1.0, step=0.5, label="Target Peak (dB)")
                n_btn = gr.Button("Normalize", variant="primary")
        n_out = gr.Audio(label="Normalized Output")

    def do_trim(path, thr, pad):
        if not path:
            return None
        x, sr = sf.read(path)
        if x.ndim == 2:
            x = np.mean(x, axis=1)

        idx = np.where(np.abs(x) > thr)[0]
        if idx.size == 0:
            return path

        start = max(0, int(idx[0]) - int((pad/1000)*sr))
        end = min(len(x), int(idx[-1]) + 1 + int((pad/1000)*sr))
        y = x[start:end]

        out_path = Path(path).parent / f"trimmed_{Path(path).name}"
        sf.write(str(out_path), y, sr)
        return str(out_path)

    def do_norm(path, peak_db):
        if not path:
            return None
        x, sr = sf.read(path)

        peak_lin = 10 ** (peak_db / 20)
        curr_peak = np.max(np.abs(x))
        if curr_peak > 0:
            y = x * (peak_lin / curr_peak)
        else:
            y = x

        out_path = Path(path).parent / f"norm_{Path(path).name}"
        sf.write(str(out_path), y, sr)
        return str(out_path)

    t_btn.click(fn=do_trim, inputs=[t_in, t_thr, t_pad], outputs=t_out)
    n_btn.click(fn=do_norm, inputs=[n_in, n_peak], outputs=n_out)
