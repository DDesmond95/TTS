#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from typing import Optional


async def run(ws_url: str, payload: dict, max_seconds: int = 60) -> None:
    try:
        import websockets  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: websockets. Install with: pip install websockets"
        )

    t0 = time.time()
    header = None
    bytes_recv = 0

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", **payload}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                bytes_recv += len(msg)
                continue
            # text message
            obj = json.loads(msg)
            if obj.get("type") == "header":
                header = obj
                print(f"[header] {header}")
            elif obj.get("type") == "end":
                t1 = time.time()
                print(f"[end] {obj}")
                print(f"[stats] seconds={t1 - t0:.2f} bytes={bytes_recv}")
                return
            else:
                print(f"[msg] {obj}")

            if time.time() - t0 > max_seconds:
                await ws.send(json.dumps({"type": "cancel"}))
                print("[warn] timeout reached; sent cancel")
                return


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal WebSocket streaming test client.")
    ap.add_argument("--ws-url", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--chunk-ms", type=int, default=60)
    args = ap.parse_args()

    payload = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "stream_format": "pcm16",
        "chunk_ms": args.chunk_ms,
    }
    asyncio.run(run(args.ws_url, payload))


if __name__ == "__main__":
    main()
