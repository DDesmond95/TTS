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

- 400 for validation errors (including invalid audio format, duration limits, unsupported sample rate/channels)
- 404 for missing model/voice
- 409 for concurrency limit reached
- 500 for runtime errors

Error body:

- { "error": { "code": "...", "message": "...", "details": {...} } }

## Voice conversion

POST /voice/convert

Inputs (recommended: multipart/form-data):

- source_audio: WAV/FLAC/MP3 upload (server will resample as needed)
- target_voice: voice profile id (type=conversion_target) or reference audio

Optional:

- output_format: "wav" | "pcm16"
- sample_rate: integer
- normalize_loudness: bool

Returns:

- JSON with run_id + artifact references + metadata
  Optionally:
- direct audio bytes if requested (small outputs)

## Singing synthesis

POST /sing/generate

Inputs:

- lyrics: string
- melody (optional, depending on supported mode):
  - midi_file upload OR
  - note list JSON OR
  - score file upload (if supported)
    Optional:
- singer_voice: voice profile id (type=singer)
- output_format, sample_rate

Returns:

- JSON with run_id + artifact references + metadata

Optionally:

- direct audio bytes if requested

## Voice editing

POST /voice/edit

Inputs (recommended: multipart/form-data):

- input_audio: WAV/FLAC/MP3 upload
- style: instruction string and/or reference audio

Optional:

- strength: float (0–1)
- output_format, sample_rate, normalize_loudness

Returns:

- JSON with run_id + artifact references + metadata
  Optionally:
- direct audio bytes if requested

## Streaming (WebSocket)

See STREAMING.md for protocol details.

TTS streaming:

- WS /ws/tts/custom_voice
- WS /ws/tts/voice_design
- WS /ws/tts/voice_clone

Voice conversion streaming:

- WS /ws/voice/live
