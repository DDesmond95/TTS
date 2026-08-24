"""Result storage and artifact management for OmniVoice Studio."""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..exceptions import AudioProcessingError, OmniVoiceError


@dataclass(frozen=True)
class RunResult:
    """Container for the output of a combined model/task run."""

    run_id: str
    run_dir: Path
    audio_path: Path | None
    sample_rate: int | None
    meta: dict[str, Any]


class OutputStore:
    """Manages local storage of task run results, artifacts, and parameters."""

    def __init__(self, outputs_dir: Path):
        """Initializes the OutputStore."""
        self.outputs_dir = outputs_dir.resolve()
        self.runs_dir = (self.outputs_dir / "runs").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def new_run_dir(self, task: str) -> tuple[str, Path]:
        """
        Create a new unique directory for an incoming task run.

        Args:
            task: The name of the task to include in the run ID.

        Returns:
            A tuple of (run_id, run_dir_path).
        """
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        # cheap monotonic suffix to avoid collisions
        suffix = f"{int(time.time() * 1000) % 100000:05d}"
        run_id = f"{ts}_{suffix}_{task}"
        run_dir = (self.runs_dir / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def write_params(self, run_dir: Path, params: dict[str, Any]) -> None:
        """
        Write the input parameters to the run directory as JSON.

        Args:
            run_dir: The directory for the run.
            params: Dictionary of parameters to save.
        """
        try:
            (run_dir / "params.json").write_text(
                json.dumps(params, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            raise OmniVoiceError(
                f"Failed to write params for run {run_dir.name}: {e}"
            ) from e

    def write_meta(self, run_dir: Path, meta: dict[str, Any]) -> None:
        """
        Write the output metadata to the run directory as JSON.

        Args:
            run_dir: The directory for the run.
            meta: Metadata dictionary to save.
        """
        try:
            (run_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception as e:
            raise OmniVoiceError(
                f"Failed to write meta for run {run_dir.name}: {e}"
            ) from e

    def save_wav(
        self, run_dir: Path, wav: np.ndarray, sr: int, filename: str = "audio.wav"
    ) -> Path:
        """
        Save a NumPy audio array to a WAV file in the run directory.

        Args:
            run_dir: The directory to save in.
            wav: The audio array.
            sr: Sample rate.
            filename: Name of the file.

        Returns:
            Path to the saved WAV file.
        """
        try:
            wav_arr = np.asarray(wav)
            out = (run_dir / filename).resolve()
            sf.write(str(out), wav_arr, sr)
            return out
        except Exception as e:
            raise AudioProcessingError(f"Failed to save WAV to {filename}: {e}") from e

    def complete_run(
        self,
        run_id: str,
        run_dir: Path,
        wavs: list[np.ndarray],
        sr: int,
        *,
        meta_extra: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Finalize a run: save all WAVs, write meta.json, and return RunResult.

        Args:
            run_id: Unique ID for the run.
            run_dir: Path to the run directory.
            wavs: List of audio arrays.
            sr: Sample rate.
            meta_extra: Additional metadata to include.

        Returns:
            A RunResult object describing the run outputs.
        """
        audio_paths = []
        for i, w in enumerate(wavs):
            fname = "audio.wav" if len(wavs) == 1 else f"audio_{i}.wav"
            audio_paths.append(self.save_wav(run_dir, w, sr, filename=fname))

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        if meta_extra:
            meta.update(meta_extra)

        self.write_meta(run_dir, meta)

        audio_path = audio_paths[0] if audio_paths else None
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    def export_run(self, run_id: str, out_zip: Path) -> Path:
        """
        Create a ZIP archive of an entire run directory.

        Args:
            run_id: ID of the run to export.
            out_zip: Destination ZIP path.

        Returns:
            Path to the created ZIP file.
        """
        src = (self.runs_dir / run_id).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Run {run_id} not found")

        try:
            with zipfile.ZipFile(out_zip, "w") as z:
                for f in src.rglob("*"):
                    z.write(f, arcname=f.relative_to(src.parent))
            return out_zip
        except Exception as e:
            raise OmniVoiceError(f"Failed to export run {run_id} as ZIP: {e}") from e
