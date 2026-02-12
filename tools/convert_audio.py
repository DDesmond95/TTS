#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)
    # data shape: (n, ch)
    return data.astype(np.float32), int(sr)


def _save_audio(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    return np.mean(x, axis=1)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    ratio = sr_out / sr_in
    n_out = int(math.floor(len(x) * ratio))
    idx = np.linspace(0, len(x) - 1, num=n_out, dtype=np.float64)
    x0 = np.floor(idx).astype(np.int64)
    x1 = np.minimum(x0 + 1, len(x) - 1)
    w = idx - x0
    return (1 - w) * x[x0] + w * x[x1]


def _normalize_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 1e-9:
        return x
    return (x / m) * peak


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert audio to standard format (mono, resample, normalize)."
    )
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument(
        "--sr", type=int, default=None, help="Target sample rate (e.g., 24000)."
    )
    ap.add_argument("--mono", action="store_true", help="Convert to mono.")
    ap.add_argument(
        "--normalize-peak", action="store_true", help="Normalize peak to ~0.95."
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    x, sr = _load_audio(inp)
    if args.mono:
        x = _to_mono(x)
    else:
        # keep channels (soundfile supports 2d), but tools assume mono often
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]

    if x.ndim == 2:
        # resample each channel if needed
        if args.sr and args.sr != sr:
            chans = []
            for c in range(x.shape[1]):
                chans.append(_resample_linear(x[:, c], sr, args.sr))
            x = np.stack(chans, axis=1)
            sr = args.sr
    else:
        if args.sr and args.sr != sr:
            x = _resample_linear(x, sr, args.sr)
            sr = args.sr

    if args.normalize_peak:
        if x.ndim == 2:
            x = np.stack([_normalize_peak(x[:, c]) for c in range(x.shape[1])], axis=1)
        else:
            x = _normalize_peak(x)

    _save_audio(out, x, sr)
    print(f"[done] wrote {out} sr={sr} shape={x.shape}")


if __name__ == "__main__":
    main()
