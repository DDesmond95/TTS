import json
import time


def check_websockets():
    try:
        import websockets  # type: ignore

        return websockets
    except ImportError:
        import sys

        print("Missing dependency: websockets. Install with: pip install websockets")
        sys.exit(1)


async def ws_run_task(ws_url: str, payload: dict):
    websockets = check_websockets()
    t0 = time.time()
    t_first = None
    t_end = None

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
        t_first = t_end or time.time()

    return t_first - t0, (t_end or time.time()) - t0
