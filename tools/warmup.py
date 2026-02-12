#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path


def warmup_local(model_id_or_path: str, task: str) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(
        f"[warmup-local] device={device} dtype={dtype} model={model_id_or_path} task={task}"
    )
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_id_or_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=None,
    )
    t1 = time.time()
    print(f"[warmup-local] load_s={t1 - t0:.2f}")

    # Tiny generation to populate caches
    if task == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text="Warmup.",
            language="English",
            speaker="Ryan",
            instruct="",
            max_new_tokens=128,
        )
    elif task == "voice_design":
        wavs, sr = model.generate_voice_design(
            text="Warmup.",
            language="English",
            instruct="Neutral voice.",
            max_new_tokens=128,
        )
    elif task == "voice_clone":
        # Requires ref_audio/ref_text; use a minimal internal sample only if user provides.
        raise SystemExit(
            "voice_clone warmup requires ref_audio and ref_text; use the API warmup instead."
        )
    else:
        raise SystemExit(f"Unknown task: {task}")

    t2 = time.time()
    print(f"[warmup-local] gen_s={t2 - t1:.2f} sr={sr} samples={len(wavs[0])}")
    print("[done] warmup complete")


def warmup_api(api_url: str, model: str) -> None:
    import requests

    print(f"[warmup-api] api_url={api_url} model={model}")
    r = requests.post(
        f"{api_url.rstrip('/')}/models/warmup", json={"model": model}, timeout=600
    )
    print(f"[warmup-api] status={r.status_code}")
    print(r.text)
    r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm up a model (local or via API).")
    ap.add_argument("--mode", choices=["local", "api"], default="local")
    ap.add_argument(
        "--model",
        required=True,
        help="HF model id or local path (local mode) / model name (api mode)",
    )
    ap.add_argument(
        "--task",
        choices=["custom_voice", "voice_design", "voice_clone"],
        default="custom_voice",
    )
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8001"))
    args = ap.parse_args()

    if args.mode == "local":
        warmup_local(args.model, args.task)
    else:
        warmup_api(args.api_url, args.model)


if __name__ == "__main__":
    main()
