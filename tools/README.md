# tools/

This folder contains standalone utility scripts. They are not imported by the runtime code under `src/`.

Conventions:

- Prefer explicit arguments over implicit defaults.
- All scripts should work whether you run the core API/UI or not.
- If a feature becomes core, move it into `src/<package>/cli.py` and keep `tools/` for one-offs.

Common env vars:

- MODELS_DIR (default: ./models)
- VOICES_DIR (default: ./voices)
- OUTPUTS_DIR (default: ./outputs)
- API_URL (default: http://localhost:8001)

Quick usage:

Model management:

- python tools/download_models.py --all --models-dir ./models
- python tools/verify_models.py --models-dir ./models
- python tools/print_system_info.py
- python tools/warmup.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

Voice assets:

- python tools/create_voice_profile.py --id my_voice --type clone
- python tools/list_voices.py --voices-dir ./voices
- python tools/build_clone_prompt.py --profile ./voices/profiles/my_voice.json --model Qwen/Qwen3-TTS-12Hz-0.6B-Base

Audio prep:

- python tools/convert_audio.py in.wav out.wav --sr 24000 --mono
- python tools/trim_silence.py in.wav out.wav
- python tools/normalize_loudness.py in.wav out.wav

Streaming tools (WebSocket):

- python tools/ws_stream_test_client.py --ws-url ws://localhost:8001/ws/tts/custom_voice --text "Hello"
- python tools/ws_stream_recorder.py --ws-url ws://localhost:8001/ws/tts/custom_voice --text "Hello" --out out.wav
- python tools/ws_stream_player.py --ws-url ws://localhost:8001/ws/tts/custom_voice --text "Hello" --device "CABLE Input"

VTuber helper:

- python tools/hotkey_tts.py --ws-url ws://localhost:8001/ws/tts/custom_voice --device "CABLE Input"

Batch/pipelines:

- python tools/bulk_generate_from_csv.py --csv prompts.csv --api-url http://localhost:8001 --task custom_voice
- python tools/srt_to_audio.py --srt captions.srt --api-url http://localhost:8001 --voice-profile narrator --out out.wav

CI/container helpers:

- python tools/export_docker_env.py --out .env
- python tools/check_docker_gpu.py
