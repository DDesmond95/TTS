# Outputs

All generated artifacts are stored under `outputs/`.

## Structure

outputs/

- runs/
  - 2026-02-12T10-22-33Z_custom_voice_ab12cd/
    - audio.wav
    - input.json
    - meta.json
  - 2026-02-12T10-30-01Z_voice_clone_ef34gh/
    - audio.wav
    - input.json
    - meta.json

## input.json (recommended)

Store the request payload:

- task name
- text(s)
- model selection
- language
- speaker/voice profile
- instruct
- generation params

## meta.json (recommended)

Store runtime results:

- sample_rate
- duration_sec
- device
- dtype
- time_to_first_audio_ms (if streaming)
- total_runtime_ms
- any warnings (e.g., truncated due to max tokens)

## Output browser behavior

UI should:

- list runs sorted by timestamp
- allow filtering by task/model/voice
- play audio
- download audio + json files
