#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _resolve_asset(base_dir: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and cache a voice clone prompt for a clone profile."
    )
    ap.add_argument(
        "--profile", required=True, help="Path to voices/profiles/<id>.json"
    )
    ap.add_argument(
        "--model",
        required=True,
        help="HF model id or local path for Base model (0.6B/1.7B)",
    )
    ap.add_argument("--voices-dir", default=os.getenv("VOICES_DIR", "./voices"))
    ap.add_argument(
        "--out-name",
        default=None,
        help="Optional output filename under voices/prompts/",
    )
    args = ap.parse_args()

    voices_dir = Path(args.voices_dir).resolve()
    profile_path = Path(args.profile).resolve()
    data = json.loads(profile_path.read_text(encoding="utf-8"))

    if data.get("type") != "clone":
        raise SystemExit("Profile type must be 'clone'.")

    clone = data.get("clone", {}) or {}
    ref_audio_path = clone.get("ref_audio_path")
    ref_text_path = clone.get("ref_text_path")
    if not ref_audio_path:
        raise SystemExit("clone.ref_audio_path is required.")
    if not ref_text_path and not bool(clone.get("x_vector_only_mode", False)):
        raise SystemExit(
            "clone.ref_text_path is required unless x_vector_only_mode=True."
        )

    ref_audio = _resolve_asset(voices_dir, ref_audio_path)
    ref_text = _resolve_asset(voices_dir, ref_text_path) if ref_text_path else None
    ref_text_str = ref_text.read_text(encoding="utf-8").strip() if ref_text else None

    import torch
    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"[info] loading model={args.model} device={device} dtype={dtype}")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        attn_implementation=None,
    )

    x_vector_only_mode = bool(clone.get("x_vector_only_mode", False))
    print(
        f"[info] building prompt ref_audio={ref_audio} x_vector_only_mode={x_vector_only_mode}"
    )

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio),
        ref_text=ref_text_str if not x_vector_only_mode else None,
        x_vector_only_mode=x_vector_only_mode,
    )

    prompts_dir = voices_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    out_name = args.out_name or f"{data.get('id', profile_path.stem)}.prompt.pt"
    out_path = (prompts_dir / out_name).resolve()

    # torch.save is simplest for arbitrary prompt structures
    torch.save(prompt_items, str(out_path))
    print(f"[done] saved prompt to {out_path}")

    # Update profile
    clone["cached_prompt_path"] = str(out_path.relative_to(voices_dir))
    data["clone"] = clone
    data.setdefault("meta", {})
    data["meta"]["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    profile_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] updated profile {profile_path}")


if __name__ == "__main__":
    main()
