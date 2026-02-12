from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    audio_path: Optional[Path]
    sample_rate: Optional[int]
    meta: Dict[str, Any]


class OutputManager:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir.resolve()
        self.runs_dir = (self.outputs_dir / "runs").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def new_run_dir(self, task: str) -> tuple[str, Path]:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        # cheap monotonic suffix to avoid collisions
        suffix = f"{int(time.time() * 1000) % 100000:05d}"
        run_id = f"{ts}_{suffix}_{task}"
        run_dir = (self.runs_dir / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def write_params(self, run_dir: Path, params: Dict[str, Any]) -> None:
        (run_dir / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def write_meta(self, run_dir: Path, meta: Dict[str, Any]) -> None:
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def save_wav(
        self, run_dir: Path, wav: np.ndarray, sr: int, filename: str = "audio.wav"
    ) -> Path:
        import soundfile as sf

        wav = np.asarray(wav)
        out = (run_dir / filename).resolve()
        sf.write(str(out), wav, sr)
        return out
