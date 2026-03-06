# Models

This project stores model weights under `models/`.

Expected layout (example):

```
models/
  qwen3/
    Qwen3-TTS-Tokenizer-12Hz/
    Qwen3-TTS-12Hz-0.6B-Base/
    Qwen3-TTS-12Hz-0.6B-CustomVoice/
    Qwen3-TTS-12Hz-1.7B-Base/
    Qwen3-TTS-12Hz-1.7B-CustomVoice/
    Qwen3-TTS-12Hz-1.7B-VoiceDesign/

  meanvc/
    checkpoints/

  tcsinger2/
    checkpoints/

  voicesculptor/
    checkpoints/
```

Notes:

- `qwen3/` contains HuggingFace-style model folders.
- `meanvc/`, `tcsinger2/`, and `voicesculptor/` typically contain checkpoint directories or model artifacts specific to those projects.
- Each model family may have different internal file structures.

The engine supports selecting models by:

- local path (preferred)
- remote model id (optional)

# Model selection policy

Default behavior:

- Prefer **0.6B Qwen3-TTS models** for stability on 6GB GPUs.
- Allow user override to **1.7B models** where possible.
- Voice conversion, singing synthesis, and voice editing models may have their own recommended checkpoints.

Engine should expose:

- `list_models()`
  scans `models/` and returns availability + capabilities

- `warmup(model_name)`
  loads the model and runs a minimal generation

# Loading policy

- Lazy load on first use
- Keep at most `MODEL_CACHE_SIZE` models in memory (default 1)
- Use `fp16` by default on GPU
- Avoid FlashAttention configuration entirely

Additional notes:

- Different model families may require different loading routines.
- The engine should isolate model-specific loading logic inside the model adapter layer.

# Download scripts

See:

- `tools/download_models.py`
- `docs/CLI.md`

for usage instructions.

# Supported model families

| Model         | Category          | Input          | Output          |
| ------------- | ----------------- | -------------- | --------------- |
| Qwen3-TTS     | text-to-speech    | text           | speech          |
| MeanVC        | voice conversion  | speech         | speech          |
| TCSinger2     | singing synthesis | lyrics / score | singing         |
| VoiceSculptor | voice editing     | speech         | modified speech |

Notes:

- **Qwen3-TTS** supports CustomVoice, VoiceDesign, VoiceClone, and tokenizer tasks.
- **MeanVC** focuses on real-time speech-to-speech voice conversion.
- **TCSinger2** generates expressive singing voice from lyrics and musical information.
- **VoiceSculptor** modifies or designs voice characteristics from reference audio.
