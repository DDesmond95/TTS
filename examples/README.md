# OmniVoice Studio Examples 🎙️

This folder contains comprehensively runnable examples to test each component of **OmniVoice Studio**.

## 🚀 Components Covered

1.  **OmniVoice Studio** (Integrated Platform API)
2.  **Qwen3-TTS** (Direct Task Usage)
3.  **MeanVC** (Voice Conversion)
4.  **TCSinger2** (Singing Synthesis)
5.  **VoiceSculptor** (Voice Design & Editing)

## 🛠️ Prerequisites

Ensure you have installed the project in editable mode:

```bash
pip install -e .
```

And download the necessary models using the CLI:

```bash
omnivoice download-models --all
```

Note: Individual components like **TCSinger2** or **MeanVC** may require additional checkpoints placed in their respective folders as described in their local READMEs.

## 🏃 Running the Example

The `test_suite.py` script provides a unified way to trigger tests for each component.

### 1. Test OmniVoice Studio (Integrated)

Basic TTS generation using the high-level Engine.

```bash
python test_suite.py --task omnivoice
```

### 2. Test Qwen3-TTS Direct

Calling the task modules directly.

```bash
python test_suite.py --task qwen3
```

### 3. Test MeanVC

Voice conversion test (requires source and target audio).

```bash
python test_suite.py --task meanvc
```

### 4. Test TCSinger2

Singing synthesis test (requires lyrics and melody config).

```bash
python test_suite.py --task tcsinger
```

### 5. Test VoiceSculptor

Voice design and editing test.

```bash
python test_suite.py --task sculptor
```

### 6. Run All Tests

```bash
python test_suite.py --task all
```

## 📁 Output

All generated audio files will be saved in the `examples/output/` directory.

## 📦 Model Locations

Individual components expect models in specific relative paths if used directly:

- **MeanVC**: `MeanVC/src/ckpt/`
- **TCSinger2**: `TCSinger2/useful_ckpts/`
- **VoiceSculptor**: Requires `llasa_3b` or similar checkpoints in local folders.

Refer to the README in each sub-folder for specific download links and setup steps for standalone usage.

---

**OmniVoice Studio** - Precision, Aesthetics, and Reliability.
