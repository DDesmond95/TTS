# Qwen3-TTS: Core Generation Functions

Qwen3-TTS exposes **three generation functions** that cover all usage patterns:

1. `generate_voice_clone`
2. `generate_custom_voice`
3. `generate_voice_design`

They are **not interchangeable**, and each has a specific purpose.

---

## 1️⃣ `generate_voice_clone` — Base TTS & Voice Cloning

### What it is

This is the **most important and most flexible function**.

It serves **two roles**:

- **Base TTS** (normal text-to-speech)
- **Voice cloning** (when reference audio is provided)

There is **no separate “generate_base” function**.
Base TTS is simply voice cloning **without a reference**.

---

### When to use it

- Normal TTS (default voices)
- Mixed-language sentences (English + Chinese + Malay)
- Cloning a voice from reference audio
- Reusing a cloned voice across many sentences

---

### Base TTS (NO reference audio)

```python
wavs, sr = model.generate_voice_clone(
    text="Hello 大家好, sekarang kita test English, 中文, dan Bahasa Melayu dalam satu ayat.",
    language="Auto",  # optional, recommended for mixed text
)
```

✔ One sentence
✔ Multiple languages
✔ No reference audio
✔ Uses the Base model behavior

---

### Voice cloning (WITH reference audio)

```python
wavs, sr = model.generate_voice_clone(
    text="I think 这个模型真的很厉害, boleh mix languages naturally.",
    language="Auto",
    ref_audio="reference.wav",
    ref_text="This is the transcript of the reference audio.",
)
```

Requirements:

- `ref_audio`: WAV path, URL, base64, or `(numpy_array, sr)`
- `ref_text`: transcript of the reference audio (unless using x-vector only mode)

---

### Reusable clone prompt (best practice)

If you want to reuse the same voice many times:

```python
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
)

wavs, sr = model.generate_voice_clone(
    text="Sentence A.",
    language="English",
    voice_clone_prompt=prompt,
)
```

This avoids recomputing speaker embeddings.

---

### Key notes

- `language="Auto"` is safest for multilingual text
- Mixed languages in **one sentence** are fully supported
- This function works with **Base models (0.6B / 1.7B)**

---

## 2️⃣ `generate_custom_voice` — Preset Speakers (CustomVoice Models)

### What it is

This function is for **CustomVoice models only**.

It uses **predefined speakers** (e.g. Vivian, Ryan, Aiden) that come with the model.

You **cannot** pass reference audio here.

---

### When to use it

- You want a **specific preset voice**
- You don’t want to record reference audio
- You want consistent voice quality per speaker

---

### Example (single sentence)

```python
wavs, sr = model.generate_custom_voice(
    text="Hello 大家好, this is a preset speaker mixing English and 中文.",
    speaker="Ryan",
    language="Auto",
    instruct="",  # optional
)
```

---

### Batch example

```python
wavs, sr = model.generate_custom_voice(
    text=[
        "Hello 大家好.",
        "Good morning everyone."
    ],
    speaker=["Vivian", "Ryan"],
    language=["Chinese", "English"],
)
```

---

### Inspect supported speakers & languages

```python
model.get_supported_speakers()
model.get_supported_languages()
```

---

### Key notes

- Only works with **`*-CustomVoice` models**
- Speaker list is fixed
- Best quality when speaker’s native language matches text
- Supports multilingual text, but accents may vary

---

## 3️⃣ `generate_voice_design` — Design a Voice by Description

### What it is

This is the **most experimental function**.

You describe the voice in **natural language**, and the model synthesizes speech in that style.

It is **not cloning** — it invents a new voice based on description.

---

### When to use it

- You want a **new character voice**
- You want emotional / stylistic control
- You plan to later turn it into a reusable cloned voice

---

### Example

```python
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="High-pitched, cute, childish female voice with exaggerated emotion.",
)
```

---

### Recommended workflow (important)

Best practice is **Design → Clone → Reuse**:

1. Use `generate_voice_design` to create a short reference clip
2. Save the output audio
3. Build a clone prompt with `create_voice_clone_prompt`
4. Use `generate_voice_clone` for all future lines

This gives consistency and better performance.
