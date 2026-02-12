#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

PROFILE_TEMPLATE = {
    "id": "",
    "type": "",  # customvoice | clone | design_template
    "display_name": "",
    "defaults": {"language": "Auto", "speaker": "", "instruct": ""},
    "clone": {
        "ref_audio_path": "",
        "ref_text_path": "",
        "x_vector_only_mode": False,
        "cached_prompt_path": "",
    },
    "design_template": {"instruct_template": "", "example_text": ""},
    "meta": {"created_at": "", "updated_at": ""},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a voice profile JSON skeleton.")
    ap.add_argument("--voices-dir", default=os.getenv("VOICES_DIR", "./voices"))
    ap.add_argument(
        "--id", required=True, help="Voice profile id (filename without .json)."
    )
    ap.add_argument(
        "--type", required=True, choices=["customvoice", "clone", "design_template"]
    )
    ap.add_argument("--display-name", default=None)
    args = ap.parse_args()

    voices_dir = Path(args.voices_dir).resolve()
    prof_dir = voices_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    prof = json.loads(json.dumps(PROFILE_TEMPLATE))
    prof["id"] = args.id
    prof["type"] = args.type
    prof["display_name"] = args.display_name or args.id
    prof["meta"]["created_at"] = now
    prof["meta"]["updated_at"] = now

    # Simplify fields based on type
    if args.type == "customvoice":
        prof.pop("clone", None)
        prof.pop("design_template", None)
    elif args.type == "clone":
        prof.pop("design_template", None)
        prof["defaults"]["speaker"] = ""
    else:
        prof.pop("clone", None)
        prof["defaults"]["speaker"] = ""

    out = prof_dir / f"{args.id}.json"
    out.write_text(
        json.dumps(prof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
