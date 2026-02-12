#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

import numpy as np


async def run(ws_url: str, payload: dict, out_path: Path) -> None:
    try:
        import websockets  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: websockets. Install with: pip install websockets"
        )

    header = None
    chunks = []

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", **payload}))

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                chunks.append(msg)
                continue

            obj = json.loads(msg)
            if obj.get("type") == "header":
                header = obj
                print(f"[header] {header}")
            elif obj.get("type") == "end":
                print(f"[end] {obj}")
                break
            else:
                print(f"[msg] {obj}")

    if not header:
        raise SystemExit("No header received; cannot decode PCM16.")

    sr = int(header["sample_rate"])
    channels = int(header.get("channels", 1))
    fmt = header.get("format", "pcm16")
    if fmt != "pcm16":
        raise SystemExit(f"Unsupported format for recorder: {fmt}")

    raw = b"".join(chunks)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    audio = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    import soundfile as sf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    print(f"[done] wrote {out_path} sr={sr} samples={len(audio)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record a WebSocket audio stream into a WAV file."
    )
    ap.add_argument("--ws-url", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--chunk-ms", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "stream_format": "pcm16",
        "chunk_ms": args.chunk_ms,
    }
    asyncio.run(run(args.ws_url, payload, Path(args.out).resolve()))


if __name__ == "__main__":
    main()
