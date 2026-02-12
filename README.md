# Qwen3-TTS Platform

Local-first Qwen3-TTS runner with:

- Python engine (`src/tts_platform/engine`)
- HTTP API (FastAPI)
- WebSocket streaming transport (PCM16 chunks)
- Web UI (Gradio)
- Voice profiles in `voices/profiles/`
- Outputs written to `outputs/runs/`

## Quickstart (local)

1. Create env

- `python -m venv .venv`
- `source .venv/bin/activate` (Linux/macOS) / `.venv\Scripts\activate` (Windows)

2. Install (CUDA Torch)

- `pip install -r requirements.txt`
- `pip install -e ".[dev]"`

3. Run API

- `tts-platform run-api --config configs/default.yaml`

4. Run UI

- `tts-platform run-ui --config configs/default.yaml`

## Folder layout

- `models/` model folders (downloaded from HF/ModelScope)
- `voices/` reusable voice profiles and reference assets
- `outputs/` generated audio + metadata
- `tools/` helper scripts (not imported by app)
