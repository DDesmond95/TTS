#!/usr/bin/env python3
import argparse
from pathlib import Path

TEMPLATE = """# Docker runtime configuration

# Paths (mounted into containers)
MODELS_DIR=./models
VOICES_DIR=./voices
OUTPUTS_DIR=./outputs

# API
API_HOST=0.0.0.0
API_PORT=8001

# UI
UI_HOST=0.0.0.0
UI_PORT=7860
UI_MODE=api
API_URL=http://localhost:8001

# Runtime
DEVICE=cuda:0
DTYPE=float16
MAX_CONCURRENT_JOBS=1
MODEL_CACHE_SIZE=1
DEFAULT_MAX_NEW_TOKENS=1024

# Optional auth
# API_KEY=change-me
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a .env template for Docker runs.")
    ap.add_argument("--out", default=".env")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.write_text(TEMPLATE, encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
