# CLI

The CLI is used for:

- running API/UI
- managing models
- quick generation tests
- streaming test client
- tokenizer utilities

Recommended commands:

Run:

- run-api --host 0.0.0.0 --port 8001
- run-ui --host 0.0.0.0 --port 7860 [--api-url ...]

Models:

- download-models [--all] [--models-dir ...]
- list-models
- warmup --model <name>

Voices:

- list-voices
- add-voice --profile <json>
- build-clone-prompt --voice-id <id>

Generate (one-shot):

- tts-custom --text "..." --speaker Vivian --language Chinese
- tts-clone --voice-id my_clone_speaker_01 --text "..."

Streaming test:

- stream-custom --text "..." ...

Tokenizer:

- encode --input <wav>
- decode --input <codes.json>

VTuber utilities:

- stream-player --ws-url <...> --device "<output device name>" [--chunk-buffer-ms 200]
- hotkey-tts --ws-url <...> --device "<output device name>" [--clipboard]
