# Troubleshooting

## CUDA not available

Symptoms:

- torch.cuda.is_available() == False

Fix:

- install CUDA-enabled torch
- verify NVIDIA driver
- verify nvidia-smi

## Out of memory (OOM)

Fix:

- use fp16
- set concurrency = 1
- reduce max_new_tokens
- prefer 0.6B models
- ensure only one model loaded

## Slow first request

Fix:

- warmup model
- keep model loaded (cache size 1)

## Streaming audio glitches

Fix:

- ensure client handles PCM16 correctly
- confirm sample rate from header
- increase chunk size (ms) slightly

## VTuber / OBS issues

No audio in OBS:

- verify OBS is capturing the virtual device (Audio Input Capture)
- confirm the bridge player is outputting to the virtual device

No mouth movement in VSeeFace:

- set microphone input to the same virtual device output/monitor
- increase input gain; confirm meter activity

Crackling:

- raise chunk_ms (e.g., 60 -> 80)
- increase client buffer (e.g., 150ms -> 250ms)
- ensure MAX_CONCURRENT_JOBS=1
