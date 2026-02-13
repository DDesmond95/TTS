#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=False)
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32), int(sr)


def save(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)


def trim(x: np.ndarray, threshold: float, pad_ms: int, sr: int) -> np.ndarray:
    if x.size == 0:
        return x
    thr = float(threshold)
    idx = np.where(np.abs(x) > thr)[0]
    if idx.size == 0:
        return x[: max(1, int(0.5 * sr))]  # keep a short chunk if all silence
    start = int(idx[0])
    end = int(idx[-1]) + 1
    pad = int((pad_ms / 1000.0) * sr)
    start = max(0, start - pad)
    end = min(len(x), end + pad)
    return x[start:end]


def main() -> None:
    ap = argparse.ArgumentParser(description="Trim leading/trailing silence (mono).")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Amplitude threshold for non-silence.",
    )
    ap.add_argument(
        "--pad-ms", type=int, default=50, help="Padding to keep around detected signal."
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    x, sr = load_mono(inp)
    y = trim(x, args.threshold, args.pad_ms, sr)
    save(out, y, sr)
    print(f"[done] wrote {out} sr={sr} in={len(x)} out={len(y)}")


if __name__ == "__main__":
    main()
