# Deployment

Supported deployment options:

1. Docker (recommended)
2. Local Python environment

## Production guidance (single 6GB GPU)

- Concurrency: 1 generation at a time
- Model cache: 1 loaded model
- Prefer 0.6B for consistent latency; enable 1.7B as an option

## Reverse proxy (optional)

If exposing publicly:

- Put API behind Nginx/Caddy
- Terminate TLS at proxy
- Forward WebSockets for streaming endpoints

## Logging

- Structured logs to stdout (Docker-friendly)
- Include request_id/run_id in logs

## VTuber deployment pattern

Recommended:

- API server: Docker (GPU enabled)
- UI: Docker or local
- Streaming bridge player: run on the host OS so it can select and play to the virtual audio device used by OBS/VSeeFace.
