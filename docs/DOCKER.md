# Docker

This project supports:

- API container
- UI container
- optional all-in-one container

## Images

1. api

- Runs FastAPI (uvicorn)
- Exposes ports: 8001
- Mounts:
  - /app/models
  - /app/voices
  - /app/outputs

2. ui

- Runs Gradio/Streamlit
- Exposes ports: 7860
- Either calls local engine (not recommended in container) or calls API

## GPU runtime

Recommended:

- NVIDIA Container Toolkit
- Run with GPU access:
  - docker run --gpus all ...

If no GPU is available:

- run CPU mode (slow) for basic testing

## VTuber note (host audio)

If you want to route streamed audio into OBS/VSeeFace on the host OS:

- run the API server in Docker (fine)
- run the streaming “bridge player” on the host OS (recommended)

Reason:

- accessing host audio devices from inside Docker is platform-specific and usually not worth the complexity for VTuber workflows.

## Example run

API:

- docker run --gpus all -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/outputs:/app/outputs \
  yourname/yourimage:tag

UI (calling API):

- docker run -p 7860:7860 \
  -e API_URL=http://host.docker.internal:8001 \
  yourname/youruiimage:tag
