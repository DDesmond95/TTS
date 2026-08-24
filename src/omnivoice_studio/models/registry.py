"""Model registry for managing and categorizing local model files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """Contains metadata about a local model."""

    name: str
    path: Path
    kind: str  # tokenizer | base | customvoice | voicedesign | unknown


class ModelRegistry:
    """
    Simple local registry: scans models_dir for subfolders and categorizes by name.
    """

    def __init__(self, models_dir: Path):
        """
        Initializes the ModelRegistry.

        Args:
            models_dir: The directory containing local models.
        """
        self.models_dir = models_dir.resolve()

    def discover(self) -> list[ModelInfo]:
        """
        Scans the models directory and returns a list of discovered models.

        Returns:
            A list of ModelInfo objects for each discovered model.
        """
        if not self.models_dir.exists() or not self.models_dir.is_dir():
            return []
        items: list[ModelInfo] = []
        for d in sorted([p for p in self.models_dir.iterdir() if p.is_dir()]):
            kind = self.infer_kind(d.name)
            items.append(ModelInfo(name=d.name, path=d, kind=kind))
        return items

    def get(self, name: str) -> ModelInfo | None:
        """
        Retrieves information about a specific model by name.

        Args:
            name: The name of the model to retrieve.

        Returns:
            A ModelInfo object if found, otherwise None.
        """
        p = (self.models_dir / name).resolve()
        if p.exists() and p.is_dir():
            return ModelInfo(name=name, path=p, kind=self.infer_kind(name))
        return None

    @staticmethod
    def infer_kind(folder_name: str) -> str:
        """
        Infers the model kind from its folder name.

        Args:
            folder_name: The name of the model folder.

        Returns:
            The inferred model kind string.
        """
        n = folder_name.lower()
        # High-priority keyword mapping
        keyword_map = [
            ("tokenizer", "tokenizer"),
            ("voicedesign", "voicedesign"),
            ("customvoice", "customvoice"),
            ("meanvc", "meanvc"),
            ("tcsinger", "tcsinger"),
            ("xcodec2", "xcodec2"),
        ]
        for key, kind in keyword_map:
            if key in n:
                return kind

        # Other patterns
        if n.endswith("-base") or "base" in n:
            return "base"
        if "voicesculptor" in n or "llasa" in n:
            return "voicesculptor"

        return "unknown"
