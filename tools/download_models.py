#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]

OPTIONAL_MODELS_17B = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def _snapshot_download(model_id: str, out_dir: Path) -> None:
    """
    Try to download using huggingface_hub.snapshot_download (preferred),
    else fall back to calling `huggingface-cli download`.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        snapshot_download = None  # type: ignore

    target = out_dir / model_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    if snapshot_download is not None:
        print(f"[download] {model_id} -> {target}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return

    # Fallback: huggingface-cli
    import subprocess

    print(f"[download] {model_id} -> {target} (via huggingface-cli)")
    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub.cli",
        "download",
        model_id,
        "--local-dir",
        str(target),
    ]
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download Qwen3-TTS models into a local models/ directory."
    )
    p.add_argument(
        "--models-dir",
        default=os.getenv("MODELS_DIR", "./models"),
        help="Local models directory.",
    )
    p.add_argument(
        "--all", action="store_true", help="Download default + 1.7B optional models."
    )
    p.add_argument(
        "--include-17b", action="store_true", help="Download 1.7B models as well."
    )
    p.add_argument(
        "--only", nargs="*", default=None, help="Download only these model ids."
    )
    args = p.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        model_ids: List[str] = list(args.only)
    else:
        model_ids = list(DEFAULT_MODELS)
        if args.all or args.include_17b:
            model_ids += OPTIONAL_MODELS_17B

    print(f"[info] models_dir = {models_dir}")
    print(f"[info] models to download = {len(model_ids)}")
    for mid in model_ids:
        _snapshot_download(mid, models_dir)

    print("[done] download complete")


if __name__ == "__main__":
    main()
