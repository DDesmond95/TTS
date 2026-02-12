from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schema import VoiceProfile


class VoiceStore:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir.resolve()
        self.profiles_dir = (self.voices_dir / "profiles").resolve()

    def list_profiles(self) -> List[VoiceProfile]:
        if not self.profiles_dir.exists():
            return []
        out: List[VoiceProfile] = []
        for p in sorted(self.profiles_dir.glob("*.json")):
            try:
                out.append(
                    VoiceProfile.model_validate_json(p.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return out

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        p = (self.profiles_dir / f"{voice_id}.json").resolve()
        if not p.exists():
            return None
        return VoiceProfile.model_validate_json(p.read_text(encoding="utf-8"))

    def resolve_path(self, rel_or_abs: str) -> Path:
        rp = Path(rel_or_abs)
        if rp.is_absolute():
            return rp
        return (self.voices_dir / rp).resolve()
