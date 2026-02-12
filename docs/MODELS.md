# Models

This project stores model weights under `models/`.

Expected layout (example):
models/

- Qwen3-TTS-Tokenizer-12Hz/
- Qwen3-TTS-12Hz-0.6B-Base/
- Qwen3-TTS-12Hz-0.6B-CustomVoice/
- Qwen3-TTS-12Hz-1.7B-Base/
- Qwen3-TTS-12Hz-1.7B-CustomVoice/
- Qwen3-TTS-12Hz-1.7B-VoiceDesign/

The engine supports selecting models by:

- local path (preferred)
- remote model id (optional)

## Model selection policy

Default behavior:

- Prefer 0.6B models for stability on 6GB.
- Allow user override to 1.7B where possible.

Engine should expose:

- list_models(): scans models/ and returns availability + capabilities
- warmup(model_name): loads and runs a minimal generation

## Loading policy

- Lazy load on first use
- Keep at most MODEL_CACHE_SIZE models in memory (default 1)
- Use fp16 by default on GPU
- Avoid FlashAttention configuration entirely

## Download scripts

See tools/download_models.py and docs/CLI.md for usage.
