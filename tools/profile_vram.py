#!/usr/bin/env python3
import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Best-effort VRAM profiling for a local model load + small generation."
    )
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument(
        "--task", choices=["custom_voice", "voice_design"], default="custom_voice"
    )
    args = ap.parse_args()

    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"torch is required: {e}") from None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. VRAM profiling requires a GPU.")

    from qwen_tts import Qwen3TTSModel

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation=None,
    )

    torch.cuda.synchronize()
    t1 = time.time()

    if args.task == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text="VRAM profile warmup.",
            language="English",
            speaker="Ryan",
            instruct="",
            max_new_tokens=128,
        )
    else:
        wavs, sr = model.generate_voice_design(
            text="VRAM profile warmup.",
            language="English",
            instruct="Neutral voice.",
            max_new_tokens=128,
        )

    torch.cuda.synchronize()
    t2 = time.time()

    peak = torch.cuda.max_memory_allocated() / (1024**2)
    reserved = torch.cuda.max_memory_reserved() / (1024**2)

    print("[timing]")
    print(f"  load_s: {t1 - t0:.2f}")
    print(f"  gen_s:  {t2 - t1:.2f}")
    print("[vram]")
    print(f"  peak_allocated_mb: {peak:.0f}")
    print(f"  peak_reserved_mb:  {reserved:.0f}")
    print(f"[audio] sr={sr} samples={len(wavs[0])}")


if __name__ == "__main__":
    main()
