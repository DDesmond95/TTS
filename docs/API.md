# API

The HTTP API exposes:

- task execution (non-stream)
- streaming generation (WebSocket)
- model and voice management helpers

Base URL example:

- http://localhost:8001

## Health

GET /health

- returns: { "status": "ok" }

GET /ready

- returns readiness, possibly includes loaded model info

## Models

GET /models

- list installed models and capabilities

POST /models/warmup

- body: { "model": "..." }

## Voices

GET /voices

- list voice profiles

POST /voices

- create/update voice profile (JSON)

## Non-stream tasks

POST /tts/custom_voice
POST /tts/voice_design
POST /tts/voice_clone
POST /tts/design_then_clone
POST /tokenizer/encode
POST /tokenizer/decode

Response pattern (recommended):

- returns JSON with:
  - run_id
  - paths or download URLs
  - metadata
    Optionally:
- direct audio bytes if requested (small outputs)

## Errors (recommended)

- 400 for validation errors
- 404 for missing model/voice
- 409 for concurrency limit reached
- 500 for runtime errors

Error body:

- { "error": { "code": "...", "message": "...", "details": {...} } }
