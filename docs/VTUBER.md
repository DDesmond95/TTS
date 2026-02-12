# VTuber / OBS integration

This document describes practical ways to use the project for VTuber workflows:

- send generated speech into OBS as a clean, mixable audio source
- feed generated speech into VTuber apps (e.g., VSeeFace) as a microphone-like input for lip sync
- minimize latency using the project’s WebSocket streaming mode

Scope:

- This is about routing and real-time usage patterns.
- No finetuning, no governance/audit.

## 1) What you can do

Common VTuber use-cases enabled by this project:

- Push-to-talk TTS: type/paste a line, hit a hotkey, voice plays immediately.
- Character switching: multiple voice profiles, switch per scene or per chat command.
- Live “agent voice”: your chat/LLM generates text continuously; TTS streams audio as it’s produced.
- Dialogue mode: multi-speaker scripts rendered line-by-line with different voice profiles.
- Lip sync: route TTS audio into a VTuber app’s microphone input so the mouth moves from audio energy.

## 2) Recommended architecture for VTuber setups

You typically want TWO separate routes:
A) Route TTS audio into OBS as its own audio source (for mixing, filters, volume).
B) Route the same TTS audio into your VTuber app as a mic input (for lip sync).

Best practice is to use a virtual audio device so you can route cleanly without feedback.

High-level signal flow:

- TTS (streaming) -> "Audio Bridge Player" -> Virtual Audio Device
- OBS captures Virtual Audio Device as an input source
- VTuber app uses Virtual Audio Device as "Microphone" input (lip sync)

The “Audio Bridge Player” is just a small client that connects to this project’s WebSocket streaming endpoint and plays audio to a chosen output device (virtual cable).

## 3) Audio routing options (by OS)

### Windows (recommended)

Two common choices:

- :contentReference[oaicite:0]{index=0} (simple “one cable” routing)
- :contentReference[oaicite:1]{index=1} (more flexible mixing/routing)

Typical setup (Virtual Cable):

1. Install Virtual Cable.
2. Set your bridge player output device to “CABLE Input”.
3. In OBS, add an “Audio Input Capture” source and select “CABLE Output”.
4. In VSeeFace, set Microphone to “CABLE Output”.

### macOS

Two common choices:

- :contentReference[oaicite:2]{index=2} (paid)
- :contentReference[oaicite:3]{index=3} (free)

Setup principle:

- Bridge player outputs to the virtual device.
- OBS captures the virtual device as input.
- VTuber app uses the same virtual device as mic input.

### Linux

Two common choices:

- :contentReference[oaicite:4]{index=4}
- :contentReference[oaicite:5]{index=5} (or PipeWire equivalent if you use it)

Setup principle:

- Create a virtual sink/source.
- Route bridge player output into that sink.
- Use the corresponding monitor/source as OBS input and VTuber mic input.

## 4) OBS integration patterns

### Pattern A: simplest (not recommended long-term)

- Play generated audio through your normal speakers/headphones.
- OBS captures “Desktop Audio”.

Downsides:

- harder to isolate TTS from other desktop sounds
- higher chance of feedback/echo

### Pattern B: recommended (clean TTS track)

- Bridge player outputs to a virtual audio device.
- OBS captures that virtual device using “Audio Input Capture”.

Benefits:

- dedicated TTS fader in OBS
- easy mute/filters (compressor, limiter)
- no contamination from game/desktop audio

Recommended OBS filters for TTS track:

- Noise suppression: usually not needed (TTS is clean)
- Compressor: mild compression to reduce jumps in loudness
- Limiter: prevent clipping when the model produces louder syllables

## 5) VSeeFace (and similar) integration for lip sync

Most VTuber apps can drive mouth movement using microphone input (audio energy / spectral cues).
To use TTS:

1. Bridge player outputs TTS audio to your virtual device.
2. In VSeeFace: set Microphone to that virtual device’s output/monitor.
3. Adjust mic gain so mouth movement is responsive but not constantly open.

Notes:

- Audio-based lip sync is not phoneme-perfect, but usually looks acceptable.
- If you hear no mouth movement:
  - verify the VTuber app is pointed at the right device
  - verify the virtual cable is receiving audio (meters)

## 6) Streaming settings for VTuber use

For low perceived latency:

- Prefer WebSocket streaming endpoints (not REST “wait for full WAV”).
- Choose a conservative chunk size:
  - chunk_ms: 40–80 ms is a good start
  - smaller chunks reduce latency but increase CPU overhead and risk crackles if the client can’t keep up
- Use PCM16 for transport (efficient, simple for most playback libraries).

Buffering rule of thumb (client side):

- buffer ~100–250 ms before starting playback
- maintain a small jitter buffer to avoid underruns (crackles)

## 7) Recommended project features to support VTuber workflows

These are “bridge utilities” that should exist either as tools/ scripts or small CLI commands.

A) Stream player (bridge)

- Connects to /ws/tts/... and plays PCM16 chunks to a selected output device.
- This is what you point at your virtual cable.

B) Hotkey TTS client

- Hotkey triggers:
  - read text from clipboard OR small input box
  - send to streaming endpoint
  - play on the configured output device

C) Preset buttons

- UI presets that populate `instruct` quickly:
  - whisper, excited, newsreader, angry, calm, comedic timing, etc.

D) Multi-voice “scene mode”

- A small UI mode that:
  - chooses voice profile A/B quickly
  - shows last 5 lines
  - has a “stop” button (cancel streaming)

## 8) Example workflows

### Workflow 1: push-to-talk TTS with lip sync

- Hotkey: speak clipboard text
- Output device: virtual cable input
- OBS: capture virtual cable output as TTS source
- VSeeFace: microphone = virtual cable output

### Workflow 2: multi-character dialogue

- Prepare voice profiles:
  - narrator (CustomVoice speaker)
  - character_1 (clone profile)
  - character_2 (clone profile)
- Use Script/Table-read pipeline:
  - input: lines with speaker tags
  - output: stitched scene audio
- Route scene audio through the same bridge device into OBS

### Workflow 3: live agent voice

- Your chatbot produces partial text
- Your server streams audio continuously
- Bridge player plays the stream into OBS and VTuber mic input

## 9) Troubleshooting checklist

No audio in OBS:

- Is OBS capturing the correct device (virtual cable output/monitor)?
- Are meters moving in OBS for that source?
- Is the bridge player output device correct?

No lip sync movement:

- Is VSeeFace microphone set to the same virtual device?
- Is the mic gain too low?
- Is the virtual cable actually receiving audio?

Crackling / stutter:

- Increase chunk_ms (e.g., 40 -> 80)
- Increase client buffer (e.g., 100ms -> 200ms)
- Ensure only one generation runs at a time (GPU concurrency = 1)

Echo / feedback:

- Do not route TTS into your physical microphone monitoring path.
- Keep TTS track separate from live mic.
