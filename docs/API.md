# 🎙️ OmniVoice Studio API Reference

The OmniVoice Studio API provides a high-performance REST interface for voice synthesis, conversion, and sculpting.

## 📡 Base URL

Default: `http://localhost:8001`

---

## 🛠️ System & Management

### Health check

`GET /health` -> `{"status": "ok"}`

### List Models

`GET /models` -> Returns a list of local models and their categories.

### List Voices

`GET /voices` -> Returns all saved voice profiles.

---

## 🎙️ Core TTS Tasks

### Custom Voice (Zero-shot)

`POST /tts/custom_voice`

- **Body**:
  ```json
  {
    "text": "Hello world",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Happy and energetic",
    "model": "optional_model_id"
  }
  ```

### Voice Design

`POST /tts/voice_design`

- **Body**: Similar to Custom Voice, used for designing new voices from text instructions.

### Voice Clone

`POST /tts/voice_clone`

- **Body**:
  ```json
  {
    "text": "Target text to speak",
    "voice_profile": "my_saved_clone_id",
    "language": "Auto"
  }
  ```

---

## 🔊 Specialized Tasks (Multi-Model)

### Voice Conversion (MeanVC)

`POST /tts/voice_conversion`

- **Body**:
  ```json
  {
    "source_audio": "path/to/source.wav",
    "target_speaker_audio": "path/to/target.wav",
    "steps": 5
  }
  ```

### Singing Synthesis (TCSinger2)

`POST /tts/singing_synthesis`

- **Body**:
  ```json
  {
    "lyrics": "I'm singing in the rain...",
    "ref_audio": "optional/path/to/style_ref.wav"
  }
  ```

### Voice Sculpting (VoiceSculptor)

`POST /tts/voice_sculpting`

- **Body**:
  ```json
  {
    "instruction": "Make the voice deeper and more whispery",
    "ref_audio": "path/to/ref.wav"
  }
  ```

---

## 📦 Response Structure

All tasks return a consistent `RunResult` object:

```json
{
  "run_id": "2024-03-07_12345_custom_voice",
  "audio_url": "/outputs/runs/2024-03-07_12345_custom_voice/audio.wav",
  "meta": {
    "sample_rate": 24000,
    "duration_sec": 3.5
  }
}
```

---

## ⚠️ Error Handling

We use standard HTTP codes and a structured error body:

| Code    | Meaning                              |
| :------ | :----------------------------------- |
| **400** | Validation or Inference error        |
| **404** | Model or Voice not found             |
| **500** | Critical loading or processing error |

**Error Response Body:**

```json
{
  "error": "ModelLoadError",
  "message": "Failed to load model: GPU OOM",
  "details": { "vram_required": "8GB" }
}
```
