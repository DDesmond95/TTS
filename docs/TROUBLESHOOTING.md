# 🛠️ OmniVoice Studio: Troubleshooting Guide

If you encounter issues, please check the **detailed error handling guide** for specific exception definitions:
👉 **[Error Handling & Troubleshooting Deep Dive](../error_handling_guide.md)**

---

## ⚡ Critical & Startup Issues

### CUDA not available

- **Symptoms**: `torch.cuda.is_available() == False` or models falling back to CPU.
- **Fix**:
  - Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  - Verify NVIDIA drivers by running `nvidia-smi` in your terminal.

### Out of Memory (OOM)

- **Fix**:
  - Use **FP16** or **BF16** in `configs/default.yaml`.
  - Set `MODEL_CACHE_SIZE: 1`.
  - Prefer **0.6B models** for GPUs with less than 8GB VRAM.
  - Disable sliding window attention if supported.

### Slow Generation / High Latency

- **Fix**:
  - Warm up the model using the CLI or UI "Model Choices" toggle.
  - Ensure your system memory isn't filled by other apps (e.g., Chrome).
  - Use **FlashAttention** if your hardware supports it (`attn_implementation: flash_attention_2`).

---

## 🎙️ Audio & Streaming Issues

### Streaming Glitches or Crackling

- **Fix**:
  - Increase the `chunk_ms` in the streaming settings (e.g., from 60 to 80).
  - Confirm the client sample rate matches the model output (usually 24000Hz or 44100Hz).
  - Reduce system load.

### No Audio in OBS / VSeeFace

- **Fix**:
  - **OBS**: Verify the "Audio Input Capture" is set to your Virtual Audio Cable output.
  - **VSeeFace**: Set "Microphone" to the same virtual device where OmniVoice is playing.
  - **Gain**: If mouth movement is absent, increase input gain in VSeeFace or the OmniVoice output volume.

---

## 📜 Error Logs

If a task fails, the **"Response / Metadata"** box in the UI will provide a detailed JSON error. Always include this when reporting issues.
