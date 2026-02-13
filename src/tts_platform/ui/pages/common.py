from __future__ import annotations

import json
from typing import Any

import requests

from ...config import AppConfig
from ...engine.engine import TTSEngine


class UIState:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mode = (cfg.ui.mode or "local_engine").strip()
        self.api_url = (cfg.ui.api_url or "http://localhost:8001").rstrip("/")
        self.engine: TTSEngine | None = None

        if self.mode == "local_engine":
            from pathlib import Path
            self.engine = TTSEngine(
                models_dir=Path(cfg.paths.models_dir),
                voices_dir=Path(cfg.paths.voices_dir),
                outputs_dir=Path(cfg.paths.outputs_dir),
                runtime=cfg.runtime,
            )

    def api_post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.api_url}{path}", json=payload, timeout=1800)
        r.raise_for_status()
        return r.json()

    def api_get(self, path: str) -> dict:
        r = requests.get(f"{self.api_url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def api_delete(self, path: str) -> dict:
        if self.mode == "local_engine":
             # This is tricky because we don't have direct access to engine.delete_voice easily here
             # without duplicating logic, but for voices it's fine.
             assert self.engine is not None
             # voice_id is the last part of path /voices/{id}
             vid = path.split("/")[-1]
             ok = self.engine.delete_voice(vid)
             return {"ok": ok}
        r = requests.delete(f"{self.api_url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def get_models(self) -> list[dict]:
        if self.mode == "local_engine":
            assert self.engine is not None
            return self.engine.list_models()
        return self.api_get("/models").get("models", [])

    def get_voices(self) -> list[dict]:
        if self.mode == "local_engine":
            assert self.engine is not None
            return self.engine.list_voices()
        return self.api_get("/voices").get("voices", [])

    def model_choices(self) -> tuple[list[str], dict[str, str]]:
        models = self.get_models()
        mapping: dict[str, str] = {}
        labels: list[str] = []
        for m in models:
            name = str(m.get("name", ""))
            kind = str(m.get("kind", "unknown"))
            path = str(m.get("path", name))
            label = f"{kind} | {name}"
            labels.append(label)
            mapping[label] = path
        labels.sort()
        return labels, mapping

    def voice_choices(self, profile_type: str | None = None) -> tuple[list[str], dict[str, str]]:
        voices = self.get_voices()
        mapping: dict[str, str] = {}
        labels: list[str] = []

        for v in voices:
            vid = str(v.get("id", ""))
            vtype = str(v.get("type", ""))
            display = str(v.get("display_name", vid))
            if profile_type and vtype != profile_type:
                continue
            label = f"{vtype} | {display} ({vid})"
            labels.append(label)
            mapping[label] = vid

        labels.sort()
        return ["(none)"] + labels, {"(none)": ""} | mapping

    @staticmethod
    def safe_json(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return str(obj)
