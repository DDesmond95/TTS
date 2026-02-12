#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List voice profiles and validate referenced assets."
    )
    ap.add_argument("--voices-dir", default=os.getenv("VOICES_DIR", "./voices"))
    args = ap.parse_args()

    voices_dir = Path(args.voices_dir).resolve()
    prof_dir = voices_dir / "profiles"
    if not prof_dir.exists():
        print(f"[info] no profiles dir found: {prof_dir}")
        return

    profiles = sorted(prof_dir.glob("*.json"))
    if not profiles:
        print("[info] no profiles found")
        return

    print(f"[info] voices_dir = {voices_dir}")
    for p in profiles:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"\n- {p.name}: INVALID JSON ({e})")
            continue

        vid = data.get("id", p.stem)
        vtype = data.get("type", "unknown")
        disp = data.get("display_name", vid)
        print(f"\n- {vid} ({vtype})  name='{disp}'  file={p.name}")

        # Validate clone assets if clone
        if vtype == "clone":
            clone = data.get("clone", {}) or {}
            ref_audio = clone.get("ref_audio_path", "")
            ref_text = clone.get("ref_text_path", "")
            cached = clone.get("cached_prompt_path", "")
            if ref_audio:
                ok = (
                    (voices_dir / ref_audio).exists()
                    if not Path(ref_audio).is_absolute()
                    else Path(ref_audio).exists()
                )
                print(f"  ref_audio: {ref_audio} [{'OK' if ok else 'MISSING'}]")
            else:
                print("  ref_audio: (not set)")
            if ref_text:
                ok = (
                    (voices_dir / ref_text).exists()
                    if not Path(ref_text).is_absolute()
                    else Path(ref_text).exists()
                )
                print(f"  ref_text:  {ref_text} [{'OK' if ok else 'MISSING'}]")
            else:
                print("  ref_text:  (not set)")
            if cached:
                ok = (
                    (voices_dir / cached).exists()
                    if not Path(cached).is_absolute()
                    else Path(cached).exists()
                )
                print(f"  cached_prompt: {cached} [{'OK' if ok else 'MISSING'}]")


if __name__ == "__main__":
    main()
