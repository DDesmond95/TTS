#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import requests

TASK_TO_ENDPOINT = {
    "custom_voice": "/tts/custom_voice",
    "voice_design": "/tts/voice_design",
    "voice_clone": "/tts/voice_clone",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk generate from CSV via REST API.")
    ap.add_argument("--csv", required=True, help="CSV with at least a 'text' column.")
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8001"))
    ap.add_argument(
        "--task", choices=list(TASK_TO_ENDPOINT.keys()), default="custom_voice"
    )
    ap.add_argument("--out-manifest", default="bulk_manifest.jsonl")
    ap.add_argument("--default-language", default="English")
    ap.add_argument("--default-speaker", default="Ryan")
    ap.add_argument("--default-instruct", default="")
    args = ap.parse_args()

    api_url = args.api_url.rstrip("/")
    endpoint = api_url + TASK_TO_ENDPOINT[args.task]

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "text" not in r:
                raise SystemExit("CSV must have a 'text' column.")
            rows.append(r)

    out_path = Path(args.out_manifest).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            payload: dict[str, Any] = {}
            payload["text"] = r["text"]
            payload["language"] = r.get("language") or args.default_language
            payload["speaker"] = r.get("speaker") or args.default_speaker
            payload["instruct"] = r.get("instruct") or args.default_instruct

            # optional: model override
            if r.get("model"):
                payload["model"] = r["model"]

            resp = requests.post(endpoint, json=payload, timeout=600)
            if resp.status_code >= 300:
                print(f"[err] row {i} status={resp.status_code} body={resp.text[:200]}")
                continue
            result = resp.json()
            out.write(
                json.dumps({"row": i, "input": r, "result": result}, ensure_ascii=False)
                + "\n"
            )
            print(f"[ok] row {i} run_id={result.get('run_id')}")

    print(f"[done] wrote manifest {out_path}")


if __name__ == "__main__":
    main()
