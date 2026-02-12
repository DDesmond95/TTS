from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    kind: str  # tokenizer | base | customvoice | voicedesign | unknown


class ModelRegistry:
    """
    Simple local registry: scans models_dir for subfolders and categorizes by name.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir.resolve()

    def scan(self) -> List[ModelInfo]:
        if not self.models_dir.exists():
            return []
        items: List[ModelInfo] = []
        for d in sorted([p for p in self.models_dir.iterdir() if p.is_dir()]):
            kind = self._infer_kind(d.name)
            items.append(ModelInfo(name=d.name, path=d, kind=kind))
        return items

    def get(self, name: str) -> Optional[ModelInfo]:
        p = (self.models_dir / name).resolve()
        if p.exists() and p.is_dir():
            return ModelInfo(name=name, path=p, kind=self._infer_kind(name))
        return None

    @staticmethod
    def _infer_kind(folder_name: str) -> str:
        n = folder_name.lower()
        if "tokenizer" in n:
            return "tokenizer"
        if "voicedesign" in n:
            return "voicedesign"
        if "customvoice" in n:
            return "customvoice"
        if n.endswith("-base") or "base" in n:
            return "base"
        return "unknown"
