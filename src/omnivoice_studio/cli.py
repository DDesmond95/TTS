"""Command-line interface for OmniVoice Studio."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn

# Ensure repo root is in path for tool imports (runtime)
_CLIP_ROOT = Path(__file__).resolve().parents[2]
if str(_CLIP_ROOT) not in sys.path:
    sys.path.append(str(_CLIP_ROOT))

try:
    from tools.download_models import (
        DEFAULT_MODELS,
        OPTIONAL_MODELS_17B,
        _snapshot_download,
    )
except ImportError:
    # Fallback for linting or environments without tools in path
    DEFAULT_MODELS = []
    OPTIONAL_MODELS_17B = []

    def _snapshot_download(*args, **kwargs):
        """No-op snapshot download fallback."""
        pass


from .api.app import create_app
from .config import load_config
from .engine.engine import TTSEngine
from .logging_utils import setup_logging
from .ui.app import create_ui

log = logging.getLogger("omnivoice_studio.cli")


def _repo_root() -> Path:
    """Get the root directory of the repository."""
    # src/omnivoice_studio/cli.py -> repo root is 3 levels up: src/omnivoice_studio -> src -> repo
    return Path(__file__).resolve().parents[2]


def cmd_run_api(args: argparse.Namespace) -> None:
    """Launch the OmniVoice Studio FastAPI server."""
    cfg = load_config(args.config)
    app = create_app(cfg, _repo_root())
    uvicorn.run(
        app, host=args.host or cfg.api.host, port=int(args.port or cfg.api.port)
    )


def cmd_run_ui(args: argparse.Namespace) -> None:
    """Launch the OmniVoice Studio Web UI."""
    cfg = load_config(args.config)
    demo = create_ui(cfg)

    # Gradio queue API differs across versions; keep GPU-safe by default.
    queue_fn = getattr(demo, "queue")
    try:
        queue_fn(default_concurrency_limit=1)
    except TypeError:
        try:
            queue_fn(concurrency_count=1)
        except TypeError:
            # oldest gradio versions: queue() exists but may accept no args
            queue_fn()

    demo.launch(
        server_name=args.host or cfg.ui.host, server_port=int(args.port or cfg.ui.port)
    )


def cmd_download_models(args: argparse.Namespace) -> None:
    """Downloads models based on the configuration and CLI flags."""
    cfg = load_config(args.config)
    m_dir = _repo_root() / cfg.paths.models_dir
    m_dir.mkdir(parents=True, exist_ok=True)

    mids = list(DEFAULT_MODELS)
    if args.include_17b:
        mids += OPTIONAL_MODELS_17B
    if args.only:
        mids = args.only

    for mid in mids:
        _snapshot_download(mid, m_dir)


def _get_engine(config_path: str) -> TTSEngine:
    """Helper to initialize the TTSEngine from a config path."""
    cfg = load_config(config_path)
    root = _repo_root()
    return TTSEngine(
        models_dir=root / cfg.paths.models_dir,
        voices_dir=root / cfg.paths.voices_dir,
        outputs_dir=root / cfg.paths.outputs_dir,
        runtime=cfg.runtime,
    )


def cmd_list_models(args: argparse.Namespace) -> None:
    """CLI command to list models in the local registry."""
    eng = _get_engine(args.config)
    models = eng.list_models()
    print(f"{'NAME':<30} {'KIND':<15} {'PATH'}")
    for m in models:
        print(f"{m['name']:<30} {m['kind']:<15} {m['path']}")


def cmd_list_voices(args: argparse.Namespace) -> None:
    """CLI command to list available voice profiles."""
    eng = _get_engine(args.config)
    voices = eng.list_voices()
    print(f"{'ID':<20} {'TYPE':<15} {'DISPLAY NAME'}")
    for v in voices:
        print(f"{v['id']:<20} {v['type']:<15} {v['display_name']}")


def cmd_synthesize(args: argparse.Namespace) -> None:
    """CLI command for one-shot TTS synthesis."""

    async def run():
        eng = _get_engine(args.config)
        res = await eng.run_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            model=args.model,
        )
        print(f"Saved to: {res.audio_path}")
        print(f"Run ID: {res.run_id}")

    asyncio.run(run())


def cmd_convert(args: argparse.Namespace) -> None:
    """CLI command for voice conversion."""

    async def run():
        eng = _get_engine(args.config)
        res = await eng.run_voice_conversion(
            source_audio=args.source,
            target_speaker_audio=args.target,
            model=args.model,
        )
        print(f"Saved to: {res.audio_path}")
        print(f"Run ID: {res.run_id}")

    asyncio.run(run())


def main() -> None:
    """Main CLI entry point."""
    setup_logging()

    p = argparse.ArgumentParser(
        prog="omnivoice", description="OmniVoice Studio - AI Voice Production Platform"
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

    p_dl = sub.add_parser("download-models", help="Download models")
    p_dl.add_argument("--config", default="configs/default.yaml")
    p_dl.add_argument("--include-17b", action="store_true")
    p_dl.add_argument("--only", nargs="*")
    p_dl.set_defaults(fn=cmd_download_models)

    p_lm = sub.add_parser("list-models", help="List local models")
    p_lm.add_argument("--config", default="configs/default.yaml")
    p_lm.set_defaults(fn=cmd_list_models)

    p_lv = sub.add_parser("list-voices", help="List voice profiles")
    p_lv.add_argument("--config", default="configs/default.yaml")
    p_lv.set_defaults(fn=cmd_list_voices)

    p_syn = sub.add_parser("synthesize", help="One-shot synthesis (TTS)")
    p_syn.add_argument("text")
    p_syn.add_argument("--config", default="configs/default.yaml")
    p_syn.add_argument("--speaker", default="Ryan")
    p_syn.add_argument("--language", default="Auto")
    p_syn.add_argument("--model", default=None)
    p_syn.set_defaults(fn=cmd_synthesize)

    p_conv = sub.add_parser("convert", help="Voice conversion (MeanVC)")
    p_conv.add_argument("source", help="Source audio path")
    p_conv.add_argument("target", help="Target speaker audio path")
    p_conv.add_argument("--config", default="configs/default.yaml")
    p_conv.add_argument("--model", default=None)
    p_conv.set_defaults(fn=cmd_convert)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
