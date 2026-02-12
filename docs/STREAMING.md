# Streaming

Streaming is provided via WebSocket.

Goal:

- low latency audio playback (client receives chunks progressively)

## WebSocket endpoints (recommended)

WS /ws/tts/custom_voice
WS /ws/tts/voice_design
WS /ws/tts/voice_clone

## Protocol

Messages:

1. Client -> Server: JSON "start" message
2. Server -> Client: JSON "header" message
3. Server -> Client: binary audio chunk frames
4. Server -> Client: JSON "end" message
5. Client -> Server: JSON "cancel" message (optional)

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
