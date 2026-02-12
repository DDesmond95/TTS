# Features

All the feature ideas for this project.

## 1. Core capabilities (what the platform must do)

- Single codebase that exposes the same functionality via:
  - Python API (importable engine/SDK)
  - HTTP API (REST)
  - Streaming API (WebSocket)
  - Web UI (Studio)
- Task-oriented design: each Qwen3-TTS capability is a “task” with a consistent interface:
  - validate → run (batch/non-stream) → stream (if supported)
- Works on a single NVIDIA GTX 1660 Ti 6GB:
  - fp16 by default
  - no FlashAttention
  - batch size defaults to 1
  - conservative max tokens defaults
  - explicit VRAM/concurrency safety checks
- Supports both local model paths (your `models/` folder) and remote model IDs (optional), but encourages local for repeatable runs.

## 2. Supported inference tasks (Qwen3-TTS feature surface)

### 2.A. Custom Voice TTS (0.6B and 1.7B)

- Single text → speech.
- Batch text → speech (list of inputs).
- Language selection:
  - explicit language, or
  - Auto language
- Speaker selection from supported premium timbres.
- Instruction control:
  - tone / emotion / style / pacing prompts (when supported by the chosen checkpoint)
- Parameter overrides per request (validated and range-limited):
  - max_new_tokens, top_p, temperature, etc.

### 2.B. Voice Design (1.7B)

- Text + natural-language “voice description” → speech.
- Batch voice design generation.
- Preset “persona templates” (e.g., “radio host”, “anime”, “calm teacher”) that compile into instructions.

### 2.C. Voice Clone (Base 0.6B and 1.7B)

- Clone from reference audio + transcript (`ref_audio` + `ref_text`) → generate new speech.
- `x_vector_only_mode` (embedding-only) option with clear warning that quality may drop.
- Reusable clone prompt builder:
  - `create_voice_clone_prompt` once, then reuse for multiple generations.
- Batch cloning: many outputs with the same clone prompt.

### 2.D. Voice Design → Clone workflow

- Design a short reference clip (VoiceDesign), then build a clone prompt from it, then generate consistent multi-line speech using Base.
- “Character pack” export:
  - persona definition (voice description)
  - reference clip
  - clone prompt bundle
  - optional default generation parameters

### 2.E. Tokenizer encode/decode (Qwen3-TTS-Tokenizer-12Hz)

- Encode:
  - audio path / URL / array → discrete codes
- Decode:
  - codes → audio
- Batch encode/decode for folders (as a utility).

## 3. API features (non-stream and streaming)

### 3.A. REST/HTTP endpoints (non-stream)

- Health + readiness:
  - `GET /health` (process up)
  - `GET /ready` (models loadable / service ready)
- Model endpoints:
  - list installed models in `models/`
  - show capabilities per model (CustomVoice/VoiceDesign/Base/Tokenizer)
  - warmup a selected model
- Voice endpoints (profiles under `voices/`):
  - list voice profiles
  - create/update/delete a voice profile (JSON)
  - optionally: build/cache clone prompt for a clone profile
- Task endpoints:
  - `POST /tts/custom_voice`
  - `POST /tts/voice_design`
  - `POST /tts/voice_clone`
  - `POST /tts/design_then_clone`
  - `POST /tokenizer/encode`
  - `POST /tokenizer/decode`
- Response options:
  - return an audio file (WAV), or
  - return a `run_id` + metadata + download reference (preferred for large outputs)
- Metadata returned (minimum):
  - sample rate, duration, model name, key parameters used

### 3.B. Streaming endpoints

- WebSocket streaming for audio:
  - server sends an initial header (sample rate, format, channels)
  - then binary audio chunks (PCM16 recommended for transport)
  - then a final message with completion metadata
- Optional HTTP chunked streaming (if you want non-WebSocket clients), noting it’s less ergonomic for binary audio.
- Client cancellation:
  - stop generation mid-stream
  - immediate resource cleanup and a final “end” frame indicating cancellation

### 3.C. API usability

- OpenAPI schema (FastAPI auto docs) with examples.
- Request validation (strict schemas for each task).
- Simple authentication option (optional but practical):
  - static API key via header, configured by env var
- Consistent error model:
  - validation errors (400)
  - missing model/voice (404)
  - concurrency busy (409) or queue behavior (configurable)
  - runtime errors (500)

## 4. Web UI features (one UI that covers everything)

- Single “Studio” UI with pages/tabs:
  - CustomVoice
  - VoiceDesign
  - VoiceClone
  - Design→Clone
  - Tokenizer
  - Pipelines (long-form, script/table-read, NPC pack, subtitles)
  - Outputs browser
  - Voices library
  - Settings / diagnostics
  - Live mode (VTuber-friendly)
- Live audio player for:
  - final WAV playback
  - streamed playback (play chunks as they arrive)
- Batch mode UI:
  - paste multiple lines
  - upload CSV/JSONL for bulk generation
- Parameter controls:
  - model selection (0.6B/1.7B variants)
  - language, speaker, instruction
  - generation params (safe ranges)
- Progress + status:
  - model loading status
  - best-effort GPU memory warning
  - queue state (queued/running/done)

## 5. Voices system (stored under `voices/`)

- “Voice profile” concept (JSON + associated reference assets):
  - CustomVoice profiles:
    - speaker ID
    - default language
    - default style instructions
  - Clone profiles:
    - reference audio(s)
    - transcript(s)
    - cached prompt artifacts (optional but recommended)
  - Design profiles (templates):
    - voice description templates
    - example reference line(s)
- Voice asset preparation utilities (UI + tools):
  - resample/convert reference audio to expected format
  - trim silence
  - normalize loudness (basic)
  - support multiple reference clips per clone profile
- Voice selection:
  - choose a profile, then generate with it
  - override profile defaults at runtime

## 6. Output management (stored under `outputs/`)

- Run folder per generation:
  - `audio.wav` (or multiple files for batch)
  - input parameters saved as JSON (reproducibility; not governance)
  - lightweight metadata (duration, sample rate, model used, timing)
- Outputs browser:
  - filter by task, model, voice profile, date
  - optional free-text note/tag for usability
- Optional export features:
  - zip selected runs
  - rename runs / add human notes

## 7. Long-form and batch pipelines (application workflows)

These are compositions of tasks + text chunking + audio stitching.

### 7.A. Long-form narrator / document reader

- Chunking strategies:
  - sentence/paragraph aware splitting
  - max character-per-chunk guard
- Stitching:
  - concatenate audio chunks with configurable silence
- Optional “style by section”:
  - headings slower, body normal, quotes with a different tone

### 7.B. Audiobook / web-novel pipeline

- Chapter ingestion (text files)
- Per-character voice mapping (CustomVoice speakers and/or clone profiles)
- Output per chapter + merged book output

### 7.C. Script / table-read generator

- Parse screenplay-like text with speaker tags
- Map tags → voice profiles
- Generate line-by-line audio with consistent voices
- Optional scene-level merge

### 7.D. NPC/dialogue pack generator

- Input: CSV/JSONL prompts with character IDs
- Output:
  - per-character folders
  - manifest file for game integration

### 7.E. Subtitle-to-speech

- Input: SRT/VTT
- Output:
  - per-caption audio
  - optional stitched track (basic timing via silence padding)

## 8. Real-time / interactive demos (streaming-first)

- “Type-to-speech live” page: as you type/submit lines, audio starts streaming quickly.
- Agent voice output mode:
  - accept partial text updates and stream continuously
- Playback barge-in behavior (client-side):
  - stop playback immediately and request cancel on the server
- Stream playback stability controls:
  - chunk size control (ms)
  - client buffer control (ms)

## 9. Tokenizer-focused utilities (practical and lightweight)

- Batch encode a dataset folder to codes (storage/transport experiments).
- Decode codes back to audio for inspection.
- “Compression explorer” UI:
  - show code length vs audio length
  - play decoded output
  - compare multiple samples quickly

## 10. Developer experience features

- Python SDK module:
  - clean, typed wrappers around API endpoints and local engine calls
  - sync and async clients
- CLI commands:
  - run-api, run-ui
  - download-models into `models/`
  - list-models, list-speakers
  - synthesize (one-shot)
  - stream test client
  - encode/decode tokenizer
  - voice profile utilities (create/list/build clone prompt)
- Config system:
  - env vars + YAML config in `configs/`
  - paths: models_dir, voices_dir, outputs_dir
  - runtime: device, dtype, conservative defaults

## 11. Performance and reliability features (1660 Ti-friendly)

- Model loading policy:
  - lazy load on first use
  - single active model by default (configurable cache size)
- VRAM safety:
  - prevent concurrent GPU-heavy jobs by default (queue/concurrency = 1)
  - optional small queue with backpressure
- CPU fallback mode (optional):
  - allow CPU run (slow) if GPU unavailable
- Determinism controls (optional):
  - set seed per request
- Warmup endpoint/UI button:
  - load model and run a tiny generation to reduce first-call latency
- Timeouts and limits:
  - request limits for max text length and max tokens
  - upload limits for reference audio size

## 12. Testing and quality gates

- Unit tests:
  - request validation
  - chunking and stitching correctness
  - file IO for outputs/voices
- Integration tests (optional if GPU not in CI):
  - mocked engine tests in CI
  - real GPU tests only on self-hosted runners (optional)
- Smoke tests:
  - minimal inference on CPU or with mocked model layer
  - API route smoke tests (no GPU required)

## 13. CI/CD and containerization (explicitly required)

### 13.A. Docker

- Dockerfile(s):
  - api image (FastAPI server)
  - ui image (Gradio/Streamlit)
  - optional all-in-one image (for local demo)
- Volume mappings:
  - `/app/models` ↔ your `models/`
  - `/app/voices` ↔ your `voices/`
  - `/app/outputs` ↔ your `outputs/`
- NVIDIA container runtime guidance (documented)
- CPU-only fallback for limited functionality/testing

### 13.B. GitHub Actions

- Lint (ruff), formatting checks, type checks (mypy), unit tests (pytest)
- Build Docker images on push/tag
- Push to Docker Hub on tags/releases (using secrets)

### 13.C. GitLab CI

- Equivalent stages:
  - lint/test
  - docker build
  - docker push to Docker Hub (or GitLab registry optionally)

## 14. Documentation set (in `docs/`)

- README.md (documentation index)
- ARCHITECTURE.md (how core/engine/tasks/api/ui fit)
- QUICKSTART.md (run locally fast)
- INSTALLATION.md (dependencies, GPU notes)
- CONFIGURATION.md (env vars + YAML config)
- MODELS.md (model layout and selection)
- VOICES.md (voice profiles and reference prep)
- OUTPUTS.md (run folders and browsing)
- TASKS.md (task definitions)
- PIPELINES.md (workflow definitions)
- API.md (REST endpoints)
- STREAMING.md (WebSocket protocol + client playback guidance)
- UI.md (Studio UI behavior and pages)
- CLI.md (commands and tools)
- DOCKER.md (containers and volume mappings)
- CI_GITHUB_ACTIONS.md (workflows and secrets)
- CI_GITLAB.md (pipelines and variables)
- DEPLOYMENT.md (production patterns)
- TROUBLESHOOTING.md (common issues)
- SECURITY.md (basic hardening)
- CHANGELOG.md (release notes)
- VTUBER.md (OBS/VSeeFace routing, live mode, bridge utilities)

## 15. VTuber / streaming integration

- Recommended routing via virtual audio device so TTS becomes:
  - a dedicated OBS audio source, and
  - a microphone-like input for VTuber apps (e.g., VSeeFace) for lip sync.
- Streaming-first operation using WebSocket:
  - PCM16 chunk streaming
  - client buffering controls
  - cancellation support
- Optional bridge utilities (tools/ or CLI):
  - stream player: plays WebSocket audio stream to a chosen output device (virtual cable)
  - hotkey TTS: hotkey reads clipboard/text and triggers streaming playback
- UI “Live mode”:
  - quick voice preset switching
  - one-click style presets (whisper/excited/etc.)
  - stop/cancel streaming button
