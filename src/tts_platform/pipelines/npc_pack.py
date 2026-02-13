from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask

log = logging.getLogger("tts_platform.pipelines.npc_pack")


class NPCLine(BaseModel):
    character_id: str
    text: str
    line_id: str


class NPCPackRequest(BaseModel):
    csv_path: str
    speaker_map: dict[str, dict[str, Any]]
    model: str | None = None
    gen: dict[str, Any] = Field(default_factory=dict)


class NPCPackPipeline:
    async def run(self, engine: Any, request: NPCPackRequest) -> Any:
        # 1. Read CSV
        lines = self._read_csv(request.csv_path)
        log.info("Processing %d lines for NPC pack", len(lines))

        run_id, run_dir = engine.outputs.new_run_dir("npc_pack")

        manifest = []

        for line in lines:
            task: Any
            task_req: Any
            speaker_config = request.speaker_map.get(line.character_id)
            if not speaker_config:
                log.warning("Unknown character_id: %s, skipping", line.character_id)
                continue

            # Create character-specific folder
            char_dir = run_dir / line.character_id
            char_dir.mkdir(parents=True, exist_ok=True)

            task_type = speaker_config.get("type", "custom_voice")
            if task_type == "custom_voice":
                task = CustomVoiceTask()
                task_req = CustomVoiceRequest(
                    text=line.text,
                    language=speaker_config.get("language", "Auto"),
                    speaker=speaker_config.get("speaker", "Ryan"),
                    instruct=speaker_config.get("instruct", ""),
                    model=request.model or speaker_config.get("model"),
                    gen=request.gen,
                )
            else:
                task = VoiceCloneTask()
                task_req = VoiceCloneRequest(
                    text=line.text,
                    language=speaker_config.get("language", "Auto"),
                    voice_profile=speaker_config.get("voice_profile"),
                    model=request.model or speaker_config.get("model"),
                    gen=request.gen,
                )

            res = await task.run(engine, task_req)

            # Move/Rename audio to the character folder
            target_name = f"{line.line_id}.wav"
            target_path = char_dir / target_name
            if res.audio_path and res.audio_path.exists():
                 target_path.write_bytes(res.audio_path.read_bytes())

            manifest.append({
                "character_id": line.character_id,
                "line_id": line.line_id,
                "text": line.text,
                "path": f"{line.character_id}/{target_name}"
            })

        # 2. Save Manifest
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        engine.outputs.write_params(run_dir, request.model_dump())

        meta = {
            "total_lines": len(manifest),
            "characters": list(request.speaker_map.keys())
        }
        engine.outputs.write_meta(run_dir, meta)

        from ..storage.outputs import RunResult
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=None, # Multiple files
            sample_rate=None,
            meta=meta,
        )

    def _read_csv(self, path: str) -> list[NPCLine]:
        p = Path(path)
        if not p.exists():
            return []

        lines: list[NPCLine] = []
        with open(p, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lines.append(NPCLine(
                    character_id=row['character_id'],
                    text=row['text'],
                    line_id=row.get('line_id', f"line_{len(lines)}")
                ))
        return lines
