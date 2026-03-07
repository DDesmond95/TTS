# Quickstart

This is the minimal path to:

- download models into `models/`
- run the API
- run the UI
- generate audio

## 1) Create environment

Recommended:

- Python 3.10–3.12
- PyTorch with CUDA enabled (matching your driver)

Example (conda):

- conda create -n tts python=3.12 -y
- conda activate tts

Install project:

- pip install -e .

## 2) Download models (local)

Use tools script:

- python tools/download_models.py --all --models-dir ./models

At minimum (typical):

- Tokenizer 12Hz
- 0.6B Base (clone)
- 0.6B CustomVoice
  Optional (if it fits your VRAM/time):
- 1.7B Base
- 1.7B CustomVoice
- 1.7B VoiceDesign

## 3) Run API

- omnivoice run-api --host 0.0.0.0 --port 8001

Check:

- GET /health
- GET /ready

## 4) Run UI

Local UI calling local engine:

- omnivoice run-ui --host 0.0.0.0 --port 7860

Or UI calling API:

- omnivoice run-ui --api-url http://localhost:8001

## 5) Generate first audio

In UI:

- Go to CustomVoice
- Enter text
- Select speaker + language
- Generate

Or via API:

- POST /tts/custom_voice

Outputs:

- Saved under outputs/runs/<timestamp>_<task>_<id>/
