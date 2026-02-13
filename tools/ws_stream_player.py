#!/usr/bin/env python3
import argparse
import asyncio
import json
import time


def list_devices() -> None:
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: sounddevice. Install with: pip install sounddevice"
        ) from None
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i:3d}: {d['name']} (max_out={d['max_output_channels']})")


async def run(
    ws_url: str, payload: dict, device: str | None, buffer_ms: int
) -> None:
    try:
        import websockets  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: websockets. Install with: pip install websockets"
        ) from None
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: sounddevice. Install with: pip install sounddevice"
        ) from None

    header = None
    q: list[bytes] = []
    started = False
    start_play_time = None

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", **payload}))

        stream = None

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                q.append(msg)
                # start playback after buffer is filled
                if header and not started:
                    sr = int(header["sample_rate"])
                    ch = int(header.get("channels", 1))
                    # compute buffered ms
                    total_bytes = sum(len(b) for b in q)
                    samples = total_bytes // 2 // ch  # int16 = 2 bytes
                    buffered_ms = int(1000 * samples / sr)
                    if buffered_ms >= buffer_ms:
                        stream = sd.RawOutputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype="int16",
                            device=device,
                            blocksize=0,
                        )
                        stream.start()
                        started = True
                        start_play_time = time.time()
                        print(f"[play] started after buffering ~{buffered_ms} ms")
                # write available chunks
                if started and stream is not None:
                    while q:
                        stream.write(q.pop(0))
                continue

            obj = json.loads(msg)
            if obj.get("type") == "header":
                header = obj
                print(f"[header] {header}")
                if obj.get("format") != "pcm16":
                    raise SystemExit(
                        f"Unsupported format for player: {obj.get('format')}"
                    )
            elif obj.get("type") == "end":
                print(f"[end] {obj}")
                break
            else:
                print(f"[msg] {obj}")

        # drain remaining queue
        if started and stream is not None:
            while q:
                stream.write(q.pop(0))
            stream.stop()
            stream.close()
            if start_play_time:
                print(f"[play] total_play_s={time.time() - start_play_time:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play a WebSocket PCM16 stream to an audio output device (VTuber/OBS bridge)."
    )
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--ws-url", help="e.g., ws://localhost:8001/ws/tts/custom_voice")
    ap.add_argument("--text", help="Text to speak")
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--chunk-ms", type=int, default=60)
    ap.add_argument(
        "--buffer-ms",
        type=int,
        default=200,
        help="Client buffer before playback to avoid crackles.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Output device name or index. Leave empty for default.",
    )
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not args.ws_url or not args.text:
        raise SystemExit("Provide --ws-url and --text, or use --list-devices.")

    payload = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "stream_format": "pcm16",
        "chunk_ms": args.chunk_ms,
    }
    asyncio.run(run(args.ws_url, payload, args.device, args.buffer_ms))


if __name__ == "__main__":
    main()
