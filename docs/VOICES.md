# Voices

All prepared voices live under `voices/`.

Goals:

- Make voice usage repeatable and convenient (not “governance”).
- Support both preset speakers (CustomVoice) and clone prompts (Base).

## Recommended structure

voices/

- profiles/
  - vivian.json
  - my_clone_speaker_01.json
- refs/
  - my_clone_speaker_01.wav
  - my_clone_speaker_01.txt
- prompts/
  - my_clone_speaker_01.prompt.json (optional cached artifacts)

## Voice profile schema (recommended)

profiles/<voice_id>.json

Fields:

- id: string (voice_id)
- type: "customvoice" | "clone" | "design_template"
- display_name: string
- defaults:
  language: "Auto" | "English" | ...
  speaker: (for customvoice) e.g., "Vivian"
  instruct: string
- clone (only if type=clone):
  ref_audio_path: path (or list of paths)
  ref_text_path: path (or list)
  x_vector_only_mode: bool
  cached_prompt_path: optional path in voices/prompts/

## Creating clone prompts

Workflow:

1. user provides ref_audio + ref_text
2. engine builds clone prompt via create_voice_clone_prompt
3. prompt saved under voices/prompts/
4. profile references cached_prompt_path

## Reference audio guidance

Best practice:

- 3–10 seconds per clip
- clean, minimal background noise
- consistent mic distance
- correct transcript for the reference clip

Utilities:

- tools/convert_audio.py to resample/normalize if needed
