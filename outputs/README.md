# outputs/

This folder stores generated audio outputs and run metadata.

Recommended run layout (created by the app/API, not manually):
outputs/

- runs/
  - 2026-02-12_001_custom_voice/
    - audio.wav
    - params.json
    - meta.json

Tips:

- Keep `outputs/` mounted as a Docker volume.
- Do not commit generated audio to git.
