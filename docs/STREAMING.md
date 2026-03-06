# Streaming

Streaming is provided via WebSocket.

Two streaming modes exist:

1. Text → audio streaming (Qwen3-TTS)
2. Audio → audio streaming (MeanVC voice conversion)

Goal:

- low latency audio playback
- real-time interaction where supported

## WebSocket endpoints (recommended)

WS /ws/tts/custom_voice
WS /ws/tts/voice_design
WS /ws/tts/voice_clone

Voice conversion streaming:

WS /ws/voice/live

## Protocol

Messages (TTS streaming):

1. Client -> Server: JSON "start" message
2. Server -> Client: JSON "header" message
3. Server -> Client: binary audio chunk frames
4. Server -> Client: JSON "end" message
5. Client -> Server: JSON "cancel" message (optional)

## Protocol: MeanVC voice conversion

MeanVC streaming requires bidirectional audio streaming.
The client continuously sends input audio frames while the server returns converted audio frames.

Message flow:

1. client -> server: start (session configuration)
2. client -> server: audio_chunk
3. server -> client: converted_audio_chunk
4. repeat steps 2–3
5. server -> client: end

### Recommended settings:

MeanVC streaming

input chunk: 20–40 ms
output chunk: 40–80 ms
sample rate: typically 16000 (recommended for VC)
alternative: 22050 depending on model

### Start message (client -> server)

JSON:

- task-specific request fields (same as REST)
- stream_format:
  - "pcm16" (recommended)
- chunk_ms:
  - desired chunk size target (server may approximate)

### Header message (server -> client)

JSON:

- type: "header"
- format: "pcm16"
- sample_rate: integer
- channels: 1

### Audio chunk frames (server -> client)

Binary payload:

- little-endian PCM16 frames
- contiguous stream of samples

### End message (server -> client)

JSON:

- type: "end"
- run_id
- duration_sec
- total_runtime_ms
- optional output path references

## Cancellation

Client can send:

- { "type": "cancel" }

Server behavior:

- stop generation ASAP
- close stream gracefully with an end frame indicating cancellation

## VTuber playback guidance (low-latency)

Recommended defaults for real-time playback:

- stream_format: pcm16
- chunk_ms: 40–80 ms (start at 60 ms)
- client buffer: 100–250 ms before starting playback

If you hear crackles/stutter:

- increase chunk_ms (more stable)
- increase client buffer
- ensure GPU concurrency is 1 (no parallel generations)
- avoid heavy background CPU load on the same machine

Tip:

- For OBS/VSeeFace routing, the streaming client should output to a virtual audio device, not to desktop speakers.
