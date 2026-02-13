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


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description="Basic RMS loudness normalization (mono).")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument(
        "--target-rms", type=float, default=0.08, help="Target RMS amplitude."
    )
    ap.add_argument(
        "--peak-limit", type=float, default=0.98, help="Clamp peak after gain."
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    x, sr = load_mono(inp)
    r = rms(x)
    if r < 1e-9:
        y = x
        gain = 1.0
    else:
        gain = float(args.target_rms) / r
        y = x * gain

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > args.peak_limit and peak > 1e-9:
        y = y * (args.peak_limit / peak)

    save(out, y, sr)
    print(f"[done] wrote {out} sr={sr} gain={gain:.3f}")


if __name__ == "__main__":
    main()
