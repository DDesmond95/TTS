#!/usr/bin/env python3
import argparse
import asyncio
import json


def _get_clipboard_text() -> str:
    # Tries pyperclip; fallback to empty
    try:
        import pyperclip  # type: ignore

        t = pyperclip.paste()
        return t.strip() if isinstance(t, str) else ""
    except Exception:
        return ""


async def speak_once(
    ws_url: str, payload: dict, device: str | None, buffer_ms: int
) -> None:
    # Reuse ws_stream_player core logic by importing it is not allowed (tools should be standalone).
    # Minimal embedded player:
    try:
        import sounddevice as sd  # type: ignore
        import websockets  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependencies: websockets and sounddevice. Install with: pip install websockets sounddevice"
        ) from None

    header = None
    q = []
    started = False
    stream = None

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", **payload}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                q.append(msg)
                if header and not started:
                    sr = int(header["sample_rate"])
                    ch = int(header.get("channels", 1))
                    total_bytes = sum(len(b) for b in q)
                    samples = total_bytes // 2 // ch
                    buffered_ms = int(1000 * samples / sr)
                    if buffered_ms >= buffer_ms:
                        stream = sd.RawOutputStream(
                            samplerate=sr, channels=ch, dtype="int16", device=device
                        )
                        stream.start()
                        started = True
                if started and stream is not None:
                    while q:
                        stream.write(q.pop(0))
                continue

            obj = json.loads(msg)
            if obj.get("type") == "header":
                header = obj
                if obj.get("format") != "pcm16":
                    raise SystemExit(f"Unsupported stream format: {obj.get('format')}")
            elif obj.get("type") == "end":
                break

    if started and stream is not None:
        while q:
            stream.write(q.pop(0))
        stream.stop()
        stream.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hotkey-like TTS loop. Works great with a virtual audio device for VTubing."
    )
    ap.add_argument("--ws-url", required=True)
    ap.add_argument(
        "--device", default=None, help="Audio output device (name or index)."
    )
    ap.add_argument("--buffer-ms", type=int, default=200)
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument(
        "--clipboard", action="store_true", help="Read text from clipboard each time."
    )
    args = ap.parse_args()

    print("[info] Press ENTER to speak. Type ':q' then ENTER to quit.")
    while True:
        line = input("> ").strip()
        if line == ":q":
            return
        if args.clipboard:
            text = _get_clipboard_text()
            if not text:
                print("[warn] clipboard empty (or pyperclip not installed).")
                continue
        else:
            text = line if line else ""
            if not text:
                print("[warn] empty text")
                continue

        payload = {
            "text": text,
            "language": args.language,
            "speaker": args.speaker,
            "instruct": args.instruct,
            "stream_format": "pcm16",
            "chunk_ms": 60,
        }
        asyncio.run(speak_once(args.ws_url, payload, args.device, args.buffer_ms))


if __name__ == "__main__":
    main()
