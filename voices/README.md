# voices/

This folder stores reusable voice assets and profiles.

Recommended layout:
voices/

- profiles/ JSON voice profiles (your main entrypoint)
- refs/ reference audio + transcripts for cloning
- prompts/ cached clone prompt artifacts (optional)

Notes:

- Profiles may reference files relative to voices/ (preferred).
- Keep reference audio clean and short (3–10 seconds).
- For clone profiles:
  - provide ref_audio_path + ref_text_path, or
  - set x_vector_only_mode=true (quality may drop).
