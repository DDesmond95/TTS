from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    models_dir: Path
    voices_dir: Path
    outputs_dir: Path
    configs_dir: Path

    @staticmethod
    def from_config(
        repo_root: Path,
        models_dir: str,
        voices_dir: str,
        outputs_dir: str,
        configs_dir: str,
    ) -> Paths:
        rr = repo_root.resolve()
        return Paths(
            repo_root=rr,
            models_dir=(
                (rr / models_dir).resolve()
                if not Path(models_dir).is_absolute()
                else Path(models_dir).resolve()
            ),
            voices_dir=(
                (rr / voices_dir).resolve()
                if not Path(voices_dir).is_absolute()
                else Path(voices_dir).resolve()
            ),
            outputs_dir=(
                (rr / outputs_dir).resolve()
                if not Path(outputs_dir).is_absolute()
                else Path(outputs_dir).resolve()
            ),
            configs_dir=(
                (rr / configs_dir).resolve()
                if not Path(configs_dir).is_absolute()
                else Path(configs_dir).resolve()
            ),
        )
