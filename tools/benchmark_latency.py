#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from statistics import mean, median


async def one_run(ws_url: str, payload: dict) -> tuple[float, float]:
    try:
        import websockets  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: websockets. Install with: pip install websockets"
        ) from None

    t0 = time.time()
    t_first = None

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", **payload}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if t_first is None:
                    t_first = time.time()
                continue
            obj = json.loads(msg)
            if obj.get("type") == "end":
                t_end = time.time()
                break

    if t_first is None:
        t_first = t_end
    return (t_first - t0), (t_end - t0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark streaming latency (TTFA and total)."
    )
    ap.add_argument("--ws-url", required=True)
    ap.add_argument("--text", default="This is a latency benchmark sentence.")
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--chunk-ms", type=int, default=60)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    payload = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "stream_format": "pcm16",
        "chunk_ms": args.chunk_ms,
    }

    ttfa: list[float] = []
    total: list[float] = []
    for i in range(args.runs):
        a, b = asyncio.run(one_run(args.ws_url, payload))
        ttfa.append(a)
        total.append(b)
        print(f"[run {i}] ttfa={a*1000:.0f} ms total={b:.2f} s")

    print("\n[summary]")
    print(f"  ttfa_ms: mean={mean(ttfa)*1000:.0f} median={median(ttfa)*1000:.0f}")
    print(f"  total_s: mean={mean(total):.2f} median={median(total):.2f}")


if __name__ == "__main__":
    main()
