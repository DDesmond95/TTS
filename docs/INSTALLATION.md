# Installation

## System requirements

Target:

- NVIDIA GTX 1660 Ti (6GB VRAM)
- CUDA-capable PyTorch build
- No FlashAttention requirement

Notes:

- Default dtype should be fp16 on this GPU.
- Keep batch size 1.
- Keep max_new_tokens conservative by default.

## Python dependencies

Project should pin:

- torch
- transformers
- qwen-tts
- soundfile (or equivalent)
- fastapi + uvicorn
- websockets (or FastAPI WebSocket support)
- gradio or streamlit
- pydantic + pydantic-settings
- ruff, mypy, pytest (dev)

## Install

Editable install:

- pip install -e ".[dev]"

GPU sanity check (example):

- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

## Common installation pitfalls

1. CUDA mismatch

- Fix by installing the correct torch wheel for your CUDA runtime.

2. OOM on model load

- Start with 0.6B models.
- Limit loaded models to 1.
- Use fp16.

3. Slow first request

- Use warmup endpoint or UI warmup action.
