#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

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

# Additional components
MEANVC_MODEL = "ASLP-lab/MeanVC"
MEANVC_PATTERNS = [
    "model_200ms.safetensors",
    "meanvc_200ms.pt",
    "fastu2++.pt",
    "vocos.pt",
]

VOICESCULPTOR_VD = "ASLP-lab/VoiceSculptor-VD"
VOICESCULPTOR_CODEC = "HKUSTAudio/xcodec2"

TCSINGER2_RELIANCE = "google/flan-t5-large"


def _snapshot_download(
    model_id: str, out_dir: Path, allow_patterns: list[str] = None
) -> None:
    """
    Try to download using huggingface_hub.snapshot_download (preferred),
    else fall back to calling `huggingface-cli download`.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print(
            "[error] huggingface_hub not installed. Please run: pip install huggingface_hub"
        )
        sys.exit(1)

    target = out_dir / model_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    print(f"[download] {model_id} -> {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download models for all OmniVoice Studio components."
    )
    p.add_argument(
        "--models-dir",
        default=os.getenv("MODELS_DIR", "./models"),
        help="Local models directory.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download models for ALL components (including 1.7B variants).",
    )
    p.add_argument(
        "--include-17b",
        action="store_true",
        help="Download Qwen3-TTS 1.7B models as well.",
    )
    p.add_argument(
        "--components",
        nargs="*",
        default=["qwen3", "meanvc", "sculptor"],
        help="Components to download models for (qwen3, meanvc, sculptor, tcsinger).",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Download only these specific HF model ids.",
    )
    args = p.parse_args()

    # Determine repo root to find specific component subdirs if needed
    _repo_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        print(f"[info] Downloading explicit model IDs to {models_dir}")
        for mid in args.only:
            _snapshot_download(mid, models_dir)
        return

    selected = set(args.components)
    if args.all:
        selected = {"qwen3", "meanvc", "sculptor", "tcsinger"}

    # Qwen3-TTS
    if "qwen3" in selected:
        q_ids = list(DEFAULT_MODELS)
        if args.all or args.include_17b:
            q_ids += OPTIONAL_MODELS_17B
        for mid in q_ids:
            _snapshot_download(mid, models_dir)

    # MeanVC
    if "meanvc" in selected:
        # Pass models_dir; _snapshot_download will append "MeanVC" automatically
        _snapshot_download(MEANVC_MODEL, models_dir, allow_patterns=MEANVC_PATTERNS)

    # VoiceSculptor
    if "sculptor" in selected:
        _snapshot_download(VOICESCULPTOR_VD, models_dir)
        _snapshot_download(VOICESCULPTOR_CODEC, models_dir)

    # TCSinger2 (Dependencies)
    if "tcsinger" in selected:
        # Route to global models/TCSinger2
        target_dir = models_dir / "TCSinger2"
        target_dir.mkdir(parents=True, exist_ok=True)
        _snapshot_download(TCSINGER2_RELIANCE, target_dir)

    print(f"[done] all selected downloads complete in {models_dir}")


if __name__ == "__main__":
    main()
