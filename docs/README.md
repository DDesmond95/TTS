# Architecture

This project is a local inference “platform” for Qwen3-TTS that provides:

- A Python service layer (core engine and task modules)
- An HTTP API (non-stream + streaming)
- A Web UI (covers all tasks and pipelines)
- Asset management for models, voices, and outputs

Constraints:

- Target GPU: NVIDIA GTX 1660 Ti 6GB (single GPU)
- Default dtype: fp16
- No FlashAttention
- No finetuning

## Top-level components

1. Core / Engine

- Loads models (lazy, cached)
- Provides task execution APIs
- Owns VRAM/concurrency limits
- Produces audio artifacts and metadata

2. Tasks

- Task modules implement each capability:
  - CustomVoice
  - VoiceDesign
  - VoiceClone (Base)
  - DesignThenClone (pipeline)
  - Tokenizer Encode/Decode

3. Pipelines

- Higher-level flows that compose tasks:
  - Long-form narration (chunk + stitch)
  - Script/table-read generator (speaker mapping)
  - NPC pack generator (bulk)
  - Subtitle-to-speech (SRT/VTT)

4. API Adapter

- FastAPI app exposes task endpoints and streaming endpoints
- Validates requests with schemas
- Maps HTTP/WebSocket to engine/task calls

5. UI Adapter

- Gradio/Streamlit UI calls the same engine (local mode) or the API (remote mode)
- Provides pages for each task and pipeline
- Includes outputs browser and voices library

## Directory semantics (runtime vs assets)

Runtime code:

- src/ All importable code
- configs/ YAML configs and defaults
- tests/ Unit/integration tests

Assets (not importable code):

- models/ Model weights (local storage)
- voices/ Prepared voices, reference clips, prompt caches
- outputs/ Generated audio runs + metadata
- docs/ This documentation
- tools/ Utility scripts (downloads, conversions, benchmarks)

## Request flow (non-stream)

Client (UI or REST) -> API schemas validate -> Task run -> Save output -> Response

Steps:

1. Resolve model selection
2. Resolve voice profile/speaker and language
3. Execute task
4. Save artifacts to outputs/
5. Return file reference or bytes + metadata

## Request flow (streaming)

Client -> WebSocket -> stream protocol -> task stream -> audio chunks -> end frame

Steps:

1. Resolve model and voice inputs
2. Start generation
3. Emit header frame (format, sample rate, channels)
4. Emit binary audio chunk frames
5. Emit end frame with metadata
6. Handle cancellation if requested

## Concurrency and VRAM policy

Default policy (safe for 6GB):

- One GPU-heavy generation at a time (queue size 1)
- Tokenizer encode/decode may be allowed concurrently (configurable)
- Model cache limited (default 1 loaded model at a time)

Rationale:

- Prevent OOM and avoid unstable latency spikes.

## Configuration ownership

- `configs/*.yaml` define defaults and per-environment settings.
- Environment variables override config file values.
- Paths are always configurable:
  - MODELS_DIR
  - VOICES_DIR
  - OUTPUTS_DIR

## Extending the platform

To add a new feature:

1. Implement a task or pipeline module
2. Add API route (non-stream and/or stream)
3. Add UI page/tab
4. Add docs entry and tests

No task should be implemented only in UI or only in API.
