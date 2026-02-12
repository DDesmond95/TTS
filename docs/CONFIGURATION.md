# Configuration

Configuration sources (highest priority wins):

1. Environment variables
2. Config file (configs/\*.yaml)
3. Built-in defaults

## Key path settings

- MODELS_DIR: default ./models
- VOICES_DIR: default ./voices
- OUTPUTS_DIR: default ./outputs

## Runtime settings

- DEVICE: "cuda:0" (default) or "cpu"
- DTYPE: "float16" (default)
- MAX_CONCURRENT_JOBS: default 1 (safe for 6GB)
- MODEL_CACHE_SIZE: default 1
- DEFAULT_MAX_NEW_TOKENS: default 1024 (adjust as needed)
- DEFAULT_TOP_P, DEFAULT_TEMPERATURE: safe defaults

## API settings

- API_HOST, API_PORT
- CORS_ORIGINS (optional)
- API_KEY (optional)

## UI settings

- UI_HOST, UI_PORT
- UI_MODE:
  - local_engine: UI calls engine directly
  - api: UI calls API via base URL

## Example config file structure

configs/default.yaml:

- paths:
  models_dir: ./models
  voices_dir: ./voices
  outputs_dir: ./outputs
- runtime:
  device: cuda:0
  dtype: float16
  max_concurrent_jobs: 1
  model_cache_size: 1
  default_max_new_tokens: 1024
- api:
  host: 0.0.0.0
  port: 8001
- ui:
  host: 0.0.0.0
  port: 7860
  mode: local_engine

## Recommended profiles

- dev.yaml: verbose logging, hot reload
- prod.yaml: conservative concurrency, structured logs
