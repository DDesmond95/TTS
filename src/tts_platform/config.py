from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    dtype: str = "float16"
    max_concurrent_jobs: int = 1
    model_cache_size: int = 1
    default_max_new_tokens: int = 1024
    default_top_p: float = 0.9
    default_temperature: float = 0.8
    text_max_chars: int = 20000
    upload_max_mb: int = 50
    seed: Optional[int] = None


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: list[str] = field(default_factory=list)
    api_key: Optional[str] = None


@dataclass
class UiConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    mode: str = "local_engine"  # local_engine | api
    api_url: str = "http://localhost:8001"


@dataclass
class PathsConfig:
    models_dir: str = "./models"
    voices_dir: str = "./voices"
    outputs_dir: str = "./outputs"
    configs_dir: str = "./configs"


@dataclass
class AppConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    api: ApiConfig
    ui: UiConfig


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Paths
    cfg.setdefault("paths", {})
    cfg["paths"]["models_dir"] = os.getenv(
        "MODELS_DIR", cfg["paths"].get("models_dir", "./models")
    )
    cfg["paths"]["voices_dir"] = os.getenv(
        "VOICES_DIR", cfg["paths"].get("voices_dir", "./voices")
    )
    cfg["paths"]["outputs_dir"] = os.getenv(
        "OUTPUTS_DIR", cfg["paths"].get("outputs_dir", "./outputs")
    )

    # Runtime
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = os.getenv(
        "DEVICE", cfg["runtime"].get("device", "cuda:0")
    )
    cfg["runtime"]["dtype"] = os.getenv("DTYPE", cfg["runtime"].get("dtype", "float16"))
    cfg["runtime"]["max_concurrent_jobs"] = int(
        os.getenv("MAX_CONCURRENT_JOBS", cfg["runtime"].get("max_concurrent_jobs", 1))
    )
    cfg["runtime"]["model_cache_size"] = int(
        os.getenv("MODEL_CACHE_SIZE", cfg["runtime"].get("model_cache_size", 1))
    )
    cfg["runtime"]["default_max_new_tokens"] = int(
        os.getenv(
            "DEFAULT_MAX_NEW_TOKENS", cfg["runtime"].get("default_max_new_tokens", 1024)
        )
    )

    # API
    cfg.setdefault("api", {})
    cfg["api"]["host"] = os.getenv("API_HOST", cfg["api"].get("host", "0.0.0.0"))
    cfg["api"]["port"] = int(os.getenv("API_PORT", cfg["api"].get("port", 8001)))
    cfg["api"]["api_key"] = os.getenv("API_KEY", cfg["api"].get("api_key"))

    # UI
    cfg.setdefault("ui", {})
    cfg["ui"]["host"] = os.getenv("UI_HOST", cfg["ui"].get("host", "0.0.0.0"))
    cfg["ui"]["port"] = int(os.getenv("UI_PORT", cfg["ui"].get("port", 7860)))
    cfg["ui"]["mode"] = os.getenv("UI_MODE", cfg["ui"].get("mode", "local_engine"))
    cfg["ui"]["api_url"] = os.getenv(
        "API_URL", cfg["ui"].get("api_url", "http://localhost:8001")
    )

    return cfg


def load_config(config_path: str | Path) -> AppConfig:
    p = Path(config_path).resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {p}")

    data = _apply_env_overrides(data)

    paths = PathsConfig(**(data.get("paths") or {}), configs_dir=str(p.parent))
    runtime = RuntimeConfig(**(data.get("runtime") or {}))

    api_dict = dict(data.get("api") or {})
    cors = api_dict.pop("cors_origins", []) or []
    api = ApiConfig(cors_origins=cors, **api_dict)

    ui = UiConfig(**(data.get("ui") or {}))

    return AppConfig(paths=paths, runtime=runtime, api=api, ui=ui)
