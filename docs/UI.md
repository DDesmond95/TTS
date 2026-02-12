# UI

The UI is a “studio” for:

- generating audio via each task
- managing voices
- browsing outputs
- running streaming demos

## Pages / tabs

1. CustomVoice

- text input (single/batch)
- language (Auto or explicit)
- speaker selection
- instruct field
- generate button
- streaming toggle

2. VoiceDesign

- text input
- language
- voice description (instruct)
- generate
- streaming toggle (if supported)

3. VoiceClone

- text input
- language
- choose voice profile OR upload ref audio + transcript
- option: x_vector_only_mode
- generate
- streaming toggle

4. Design -> Clone

- voice design config (ref text + description)
- clone targets (one or many lines)
- outputs: reference + generated lines

5. Tokenizer

- encode: upload audio/url
- decode: paste codes/upload json
- play decoded audio

6. Voices library

- list profiles
- create/edit profile
- upload reference clips
- precompute clone prompt

7. Outputs browser

- list runs
- filters
- play + download

8. Settings / diagnostics

- show device, dtype, loaded model
- warmup buttons
- concurrency queue status

## UI modes

- local engine mode: calls Python engine directly
- api mode: calls HTTP API

Prefer api mode when running in Docker.

## Live mode (VTuber-friendly)

A dedicated “Live mode” page is recommended for streaming use:

- single-line input + history of recent lines
- voice profile quick switch (A/B or dropdown)
- style preset buttons (fills instruct quickly)
- streaming playback enabled by default
- big STOP button (sends cancel to WebSocket)
- optional “speak clipboard” action (paired with a hotkey tool)
