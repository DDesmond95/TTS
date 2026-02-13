#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import requests


def parse_srt(text: str) -> list[str]:
    # Minimal SRT parsing: extract caption text blocks
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    lines: list[str] = []
    for b in blocks:
        parts = b.strip().splitlines()
        if len(parts) < 3:
            continue
        caption = " ".join(p.strip() for p in parts[2:] if p.strip())
        if caption:
            lines.append(caption)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simple subtitle-to-speech: SRT -> generate per-caption and stitch (server-side preferred)."
    )
    ap.add_argument("--srt", required=True)
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8001"))
    ap.add_argument(
        "--voice-profile",
        default=None,
        help="Voice profile id if your API supports it.",
    )
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional: if API returns a run with stitched file, store ref here.",
    )
    args = ap.parse_args()

    srt_path = Path(args.srt).resolve()
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    captions = parse_srt(text)
    if not captions:
        raise SystemExit("No captions parsed from SRT.")

    # Best approach is to have a pipeline endpoint. If not available, this tool can just batch-call.
    # Here we do batch call to /tts/custom_voice with list text if supported.
    api_url = args.api_url.rstrip("/")
    endpoint = api_url + "/tts/custom_voice"

    payload = {
        "text": captions,
        "language": [args.language] * len(captions),
        "speaker": [args.speaker] * len(captions),
        "instruct": [args.instruct] * len(captions),
    }
    if args.voice_profile:
        payload["voice_profile"] = args.voice_profile

    resp = requests.post(endpoint, json=payload, timeout=1800)
    print(f"[status] {resp.status_code}")
    print(resp.text[:400])
    resp.raise_for_status()

    result = resp.json()
    print("[done] request complete")
    if args.out:
        Path(args.out).write_text(str(result), encoding="utf-8")


if __name__ == "__main__":
    main()
