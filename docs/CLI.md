# ⌨️ OmniVoice Studio CLI Reference

The unified `omnivoice` command is your primary tool for managing the studio, processing audio, and running servers.

## 🚀 Server Commands

### Run Web Studio UI

`omnivoice run-ui [--host 0.0.0.0] [--port 7860] [--config configs/default.yaml]`

### Run HTTP API

`omnivoice run-api [--host 0.0.0.0] [--port 8001] [--config configs/default.yaml]`

---

## 📦 Model & Voice Management

### List Local Models

`omnivoice list-models` -- Scans your models directory and displays categories.

### List Voice Profiles

`omnivoice list-voices` -- Displays all saved voice clones and templates.

### Download Models

`omnivoice download-models [--include-17b] [--only model_id]`

---

## 🎙️ Inference Tasks (One-Shot)

### Text-to-Speech (Qwen3-TTS)

`omnivoice synthesize "Hello from the CLI!" [--speaker Ryan] [--language Auto]`

### Voice Conversion (MeanVC)

`omnivoice convert path/to/source.wav path/to/target.wav [--model meanvc_checkpoints]`

---

## 🛠️ Advanced Usage

### Global Options

- `--config`: Path to your YAML configuration file (default: `configs/default.yaml`).

### Tokenizer Utilities

- `omnivoice encode path/to/audio.wav` -- Convert audio to discrete codes.
- `omnivoice decode path/to/codes.json` -- Reconstruct audio from codes.

---

## 📜 Logs & Debugging

The CLI automatically routes logs to the console. Use environment variables for more control:

- `LOG_LEVEL=DEBUG omnivoice run-ui`
