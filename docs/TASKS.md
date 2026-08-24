# Tasks

A “task” is an inference capability with a stable interface.

Common interface:

- `validate(request) -> request`
- `run(request) -> result (non-stream)`
- `stream(request) -> async iterator of audio chunks (if supported)`

Tasks may support **batch inference**, **streaming inference**, or both depending on the underlying model.

# Task: CustomVoice

Inputs:

- `text`: string | list[string]
- `language`: string | list[string] (or `"Auto"`)
- `speaker`: string | list[string]
- `instruct`: optional string | list[string]
- generation kwargs:
  - `max_new_tokens`
  - `top_p`
  - `temperature`
  - etc.

Outputs:

- `wav(s)`
- `sample_rate`

Streaming support:

- yes

# Task: VoiceDesign (1.7B)

Inputs:

- `text`
- `language`
- `instruct` (voice description)

Outputs:

- `wav`
- `sample_rate`

Streaming support:

- yes

# Task: VoiceClone (Base)

Inputs:

- `text`
- `language`

Reference voice options:

- `ref_audio` (path / URL / array)
- `ref_text` (string)

or

- `voice_clone_prompt` (cached prompt object)

Options:

- `x_vector_only_mode`

Outputs:

- `wav(s)`
- `sample_rate`

Streaming support:

- yes

# Task: DesignThenClone

Pipeline:

1. VoiceDesign generates a reference clip
2. Create voice clone prompt from that clip
3. VoiceClone generates target lines

Outputs:

- reference clip artifact
- clone outputs

Streaming support:

- optional (depends on underlying clone task)

# Task: Tokenizer Encode/Decode

Encode inputs:

- audio path / URL / array

Decode inputs:

- codes from encode output

Outputs:

- codes (encode)
- wav (decode)

Streaming support:

- no

# Task: VoiceConversion (MeanVC)

Voice conversion transforms **input speech into a target voice** while preserving the linguistic content.

Inputs:

- `source_audio`: Path or URL to source recording.
- `target_speaker_audio`: Path or URL to target identity recording.
- `steps`: DDIM sampling steps (default: 5).

Outputs:

- converted audio waveform
- `sample_rate`

---

# Task: SingingSynthesis (TCSinger2)

Singing synthesis generates expressive singing audio from lyrics and optional reference audio.

Inputs:

- `lyrics`: Text to sing.
- `ref_audio`: Optional path to a reference singer for style/timbre guidance.

Outputs:

- singing audio waveform
- `sample_rate`

---

# Task: VoiceSculpt (VoiceSculptor)

Voice editing/sculpting modifies the timbre or style of an existing speech recording using natural language instructions.

Inputs:

- `ref_audio`: Path to the input audio to be modified.
- `instruction`: Natural language description of the desired change.

Outputs:

- edited speech audio waveform
- `sample_rate`
