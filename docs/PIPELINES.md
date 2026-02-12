# Pipelines

Pipelines are higher-level workflows composed of tasks plus text/audio utilities.

## Long-form narration

Purpose:

- Convert long text into a single continuous audio file

Steps:

1. Split input into chunks (paragraph/sentence-based)
2. Generate per-chunk audio via CustomVoice or Clone
3. Stitch audio with configurable silence

Outputs:

- one merged WAV
- per-chunk artifacts (optional)

## Audiobook / web-novel

Inputs:

- multiple chapter files

Features:

- per-character voice mapping (profiles)
- per-chapter exports
- optional full-book merge

## Script / table-read

Inputs:

- script text with speaker tags

Features:

- map tag -> voice profile
- generate line-by-line
- optional per-scene merge

## NPC pack generator

Inputs:

- CSV/JSONL rows:
  - character_id, text, optional style

Outputs:

- folder per character
- manifest json for integration

## Subtitle-to-speech (SRT/VTT)

Inputs:

- SRT/VTT file

Features:

- generate per-caption audio
- stitch into continuous track (optional)
- preserve caption timing (optional; basic silence padding)
