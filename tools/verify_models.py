#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_SUBDIRS = [
    "Qwen3-TTS-Tokenizer-12Hz",
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice",
]

OPTIONAL_SUBDIRS = [
    "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def dir_size_bytes(path: Path) -> int:
    total = 0
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
    ap = argparse.ArgumentParser(description="Verify local models directory contents.")
    ap.add_argument("--models-dir", default=os.getenv("MODELS_DIR", "./models"))
    ap.add_argument(
        "--require-17b", action="store_true", help="Fail if 1.7B models not present."
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    print(f"[info] models_dir = {models_dir}")

    required = REQUIRED_SUBDIRS + (OPTIONAL_SUBDIRS if args.require_17b else [])
    missing: List[str] = []
    present: List[Tuple[str, str]] = []

    for name in required:
        d = models_dir / name
        if not d.exists():
            missing.append(name)
        else:
            present.append((name, human(dir_size_bytes(d))))

    if present:
        print("\n[present]")
        for name, size in present:
            print(f"  - {name}: {size}")

    if not args.require_17b:
        print("\n[optional]")
        for name in OPTIONAL_SUBDIRS:
            d = models_dir / name
            print(f"  - {name}: {'OK' if d.exists() else 'missing'}")

    if missing:
        print("\n[missing required]")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(2)

    print("\n[done] required models present")


if __name__ == "__main__":
    main()
