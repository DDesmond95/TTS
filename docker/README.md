# Docker

Build locally:

- `docker build -f docker/Dockerfile.api -t qwen3-tts-api .`
- `docker build -f docker/Dockerfile.ui  -t qwen3-tts-ui  .`

Run via compose:

- `docker compose -f docker/docker-compose.yml up --build`

Volumes (recommended):

- `./models:/app/models`
- `./voices:/app/voices`
- `./outputs:/app/outputs`
- `./configs:/app/configs`

GPU:

- Install NVIDIA Container Toolkit on the host.
- Start containers with GPU enabled (method depends on your Docker/Compose version).
