from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .api.app import create_app
from .config import load_config
from .logging_utils import setup_logging
from .ui.app import create_ui

log = logging.getLogger("tts_platform.cli")


def _repo_root() -> Path:
    # src/tts_platform/cli.py -> repo root is 3 levels up: src/tts_platform -> src -> repo
    return Path(__file__).resolve().parents[2]


def cmd_run_api(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    app = create_app(cfg, _repo_root())
    uvicorn.run(
        app, host=args.host or cfg.api.host, port=int(args.port or cfg.api.port)
    )


def cmd_run_ui(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    demo = create_ui(cfg)
    demo.queue(concurrency_count=1)  # keep GPU-safe by default
    demo.launch(
        server_name=args.host or cfg.ui.host, server_port=int(args.port or cfg.ui.port)
    )


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser(
        prog="tts_platform", description="Qwen3-TTS local platform"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_api = sub.add_parser("run-api", help="Run FastAPI server")
    p_api.add_argument("--config", default="configs/default.yaml")
    p_api.add_argument("--host", default=None)
    p_api.add_argument("--port", default=None)
    p_api.set_defaults(fn=cmd_run_api)

    p_ui = sub.add_parser("run-ui", help="Run Studio UI")
    p_ui.add_argument("--config", default="configs/default.yaml")
    p_ui.add_argument("--host", default=None)
    p_ui.add_argument("--port", default=None)
    p_ui.set_defaults(fn=cmd_run_ui)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
