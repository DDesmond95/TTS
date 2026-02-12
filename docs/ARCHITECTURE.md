# Architecture

This project is a local inference platform for Qwen3-TTS that exposes a single, consistent capability surface through:

- Python service layer (engine + task modules)
- HTTP API (REST for non-stream)
- Streaming API (WebSocket for chunked audio)
- Web UI (studio for all tasks and pipelines)

Design goals:

- One codepath for inference (API and UI must call the same core task implementations).
- Conservative runtime defaults for a single 6GB GPU (GTX 1660 Ti):
  - fp16 by default
  - no FlashAttention
  - batch size 1 by default
  - concurrency limited (default 1 GPU job at a time)
- Simple local asset layout:
  - models/ for weights
  - voices/ for reusable voice profiles and reference assets
  - outputs/ for generated audio artifacts

Non-goals:

- No finetuning.
- No governance/audit subsystems.

## 1. Top-level system overview

The system is split into three layers:

A) Core layer (importable runtime code)

- Responsible for: configuration, model loading, task execution, audio IO, output writing.

B) Adapter layer (interfaces)

- API adapter: FastAPI routes and request/response schemas.
- UI adapter: Gradio/Streamlit pages, calling either local engine or the API.

C) Asset layer (data on disk)

- models/: model weights and tokenizer
- voices/: voice profiles + reference audio + cached prompts
- outputs/: generated results

High-level dependency direction:

- adapters -> core -> (assets)
- core must not import adapters.

## 2. Code organization (module responsibilities)

Recommended module groups inside src/<package>/:

1. core/

- settings.py
  - reads configs and env vars
  - validates required paths
- paths.py
  - resolves and creates directories for models/voices/outputs
- logging.py
  - consistent logging format and request/run ids
- exceptions.py
  - shared exception types for consistent error handling

2. engine/

- model_registry.py
  - maps “model names” to local paths
  - declares model capabilities per checkpoint (custom, clone, design, tokenizer)
- loader.py
  - lazy load models with dtype/device options
  - caching policy (default one loaded model)
- audio_io.py
  - write wav
  - basic audio sanity utilities (optional normalize/resample)
- streaming.py
  - chunking strategy for streaming transport
  - async generator wrappers
- prompt_cache.py
  - build and reuse clone prompts
  - optional persist prompt artifacts to voices/prompts/

3. tasks/

- base.py
  - Task interface (validate/run/stream)
- custom_voice.py
- voice_design.py
- voice_clone.py
- design_then_clone.py
- tokenizer_codec.py

4. pipelines/

- long_form.py
  - text chunking + per-chunk generation + stitch
- script_table_read.py
  - parse speaker-tag scripts and map to voice profiles
- npc_pack.py
  - batch generate from CSV/JSONL
- subtitles.py
  - SRT/VTT parse + generate per caption + optional stitch

5. voices/

- schemas.py
  - VoiceProfile data model
- store.py
  - CRUD on voices/profiles/\*.json
  - handles refs and prompt paths
- utils.py
  - helper: hash ref audio, validate transcript exists

6. storage/

- outputs.py
  - write a run folder
  - enumerate runs for UI/API browsing
- naming.py
  - run id and folder naming policy

7. api/

- app.py
  - create_app() wiring, middleware, dependency injection
- schemas/
  - request/response models for all endpoints
- routes/
  - health.py, models.py, voices.py, tasks.py, stream.py

8. ui/

- app.py
  - UI entry point and routing/tabs
- pages/
  - one page per task/pipeline
- state.py
  - shared UI state (selected model, voice profile)

## 3. Asset layout (disk structure)

models/

- local checkpoint folders (downloaded by tools scripts)

voices/

- profiles/
  - \*.json voice profiles
- refs/
  - reference audio + transcript files
- prompts/
  - optional cached clone prompt artifacts

outputs/

- runs/
  - <timestamp>_<task>_<run_id>/
    - audio.wav (or multiple)
    - input.json
    - meta.json

tools/

- download_models.py
- convert_audio.py
- benchmark.py
- verify_models.py

## 4. Request lifecycle (REST / non-stream)

Typical non-stream request flow:

1. API/UI receives request
2. Validate request schema
3. Resolve model and voice
   - model_registry chooses the checkpoint
   - voice store returns voice profile defaults, if used
4. Engine loads model if needed (lazy load)
5. Task.run() executes inference
6. Storage writes outputs run folder
7. Response returns run_id and download reference (or direct WAV)

Sequence diagram (conceptual):

Client
-> API/UI
-> Task.validate()
-> Engine.resolve_model()
-> Engine.get_model() (lazy load)
-> Task.run()
-> Storage.write_run()
<- Response(run_id, metadata, artifact refs)

## 5. Request lifecycle (WebSocket streaming)

Streaming is implemented as:

- JSON control messages + binary audio chunk frames.

Flow:

1. Client opens WebSocket to /ws/...
2. Client sends JSON start message containing the same fields as the REST request
3. Server validates request and resolves model/voice
4. Server loads model if needed
5. Server sends JSON header frame: audio format + sample rate + channels
6. Server emits binary PCM16 chunks until complete
7. Server sends JSON end frame: run_id + timing metadata + artifact references
8. Client may send cancel message; server stops and sends an end frame marked cancelled

Sequence diagram (conceptual):

Client
-> WS connect
-> {type:"start", ...}
Server
-> {type:"header", sr, format:"pcm16", channels:1}
-> <binary chunk>
-> <binary chunk>
-> {type:"end", run_id, ...}
Client
<- close

## 6. Concurrency model (single 6GB GPU)

Default safe policy:

- MAX_CONCURRENT_JOBS = 1 for GPU-heavy tasks
- Additional requests return 409 (busy) or wait in a small in-process queue (configurable)

Model caching:

- MODEL_CACHE_SIZE = 1 by default
- LRU eviction if you allow >1

Rationale:

- Avoid OOM and keep latency stable.

## 7. Model selection policy

The registry should expose:

- “available models” (what exists in models/)
- “capabilities” per model (custom/design/clone/tokenizer)
- “defaults” (recommended model per task)

UI and API accept:

- explicit model override per request
- fallback to default for the task

## 8. Extending the system

Add a new feature by adding it as:

- a new task (if it is a single inference capability), or
- a new pipeline (if it composes multiple tasks)

Steps:

1. Add module under tasks/ or pipelines/
2. Add API endpoint(s) under api/routes/
3. Add UI page under ui/pages/
4. Add docs section (TASKS.md or PIPELINES.md)
5. Add tests (at least validation + basic non-GPU logic)

## 9. Operational boundaries

- The core engine must remain importable without starting servers.
- The UI must not implement inference logic directly.
- Tools scripts must not be imported by runtime modules (tools/ is separate by design).
