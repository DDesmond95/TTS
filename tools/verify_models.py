#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# Package-level models
QWEN3_REQUIRED = [
    "Qwen3-TTS-Tokenizer-12Hz",
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice",
]

QWEN3_OPTIONAL = [
    "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

SCULPTOR_REQUIRED = [
    "VoiceSculptor-VD",
    "xcodec2",
]

# Component-specific models (inside models dir)
MEANVC_REQUIRED_FILES = [
    "MeanVC/model_200ms.safetensors",
    "MeanVC/vocos.pt",
]


def dir_size_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify local models for all OmniVoice Studio components."
    )
    ap.add_argument("--models-dir", default=os.getenv("MODELS_DIR", "./models"))
    ap.add_argument(
        "--require-17b", action="store_true", help="Fail if 1.7B models not present."
    )
    args = ap.parse_args()

    _repo_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir).resolve()
    print(f"[info] models_dir = {models_dir}")

    all_ok = True

    # 1. Qwen3-TTS
    print("\n[Qwen3-TTS]")
    qwen_list = QWEN3_REQUIRED + (QWEN3_OPTIONAL if args.require_17b else [])
    for name in qwen_list:
        d = models_dir / name
        if not d.exists():
            print(f"  [MISSING] {name}")
            all_ok = False
        else:
            print(f"  [OK] {name} ({human(dir_size_bytes(d))})")

    # 2. VoiceSculptor
    print("\n[VoiceSculptor]")
    for name in SCULPTOR_REQUIRED:
        d = models_dir / name
        if not d.exists():
            print(f"  [MISSING] {name}")
            all_ok = False
        else:
            print(f"  [OK] {name} ({human(dir_size_bytes(d))})")

    # 3. MeanVC
    print("\n[MeanVC]")
    for rel_path in MEANVC_REQUIRED_FILES:
        p = models_dir / rel_path
        if not p.exists():
            print(f"  [MISSING] {rel_path} (expected in {models_dir})")
            all_ok = False
        else:
            print(f"  [OK] {rel_path} ({human(dir_size_bytes(p))})")

    if not all_ok:
        print("\n[error] some required models are missing.")
        raise SystemExit(1)

    print("\n[done] all required models present for the selected profile.")


if __name__ == "__main__":
    main()
