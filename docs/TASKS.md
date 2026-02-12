# Tasks

A “task” is an inference capability with a stable interface.

Common interface:

- validate(request) -> request
- run(request) -> result (non-stream)
- stream(request) -> async iterator of audio chunks (if supported)

## Task: CustomVoice

Inputs:

- text: string | list[string]
- language: string | list[string] (or "Auto")
- speaker: string | list[string]
- instruct: optional string | list[string]
- generation kwargs: max_new_tokens, top_p, temperature, etc.

Outputs:

- wav(s), sample_rate

## Task: VoiceDesign (1.7B)

Inputs:

- text
- language
- instruct (voice description)

Outputs:

- wav, sample_rate

## Task: VoiceClone (Base)

Inputs:

- text
- language
- ref_audio (path/url/array)
- ref_text (string)
  or:
- voice_clone_prompt (cached prompt object)
  Options:
- x_vector_only_mode

Outputs:

- wav(s), sample_rate

## Task: DesignThenClone

Pipeline:

1. VoiceDesign generates a reference clip
2. Create voice clone prompt from that clip
3. VoiceClone generates target lines

Outputs:

- reference clip artifact
- clone outputs

## Task: Tokenizer Encode/Decode

Encode inputs:

- audio path/url/array

Decode inputs:

- codes from encode output

Outputs:

- codes (encode)
- wav (decode)
