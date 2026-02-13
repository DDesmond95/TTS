from __future__ import annotations

from pathlib import Path

from .schema import VoiceProfile


class VoiceStore:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir.resolve()
        self.profiles_dir = (self.voices_dir / "profiles").resolve()

    def list_profiles(self) -> list[VoiceProfile]:
        if not self.profiles_dir.exists():
            return []
        out: list[VoiceProfile] = []
        for p in sorted(self.profiles_dir.glob("*.json")):
            try:
                out.append(
                    VoiceProfile.model_validate_json(p.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return out

    def get(self, voice_id: str) -> VoiceProfile | None:
        p = (self.profiles_dir / f"{voice_id}.json").resolve()
        if not p.exists():
            return None
        return VoiceProfile.model_validate_json(p.read_text(encoding="utf-8"))

    def save(self, voice_id: str, profile: VoiceProfile) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        p = (self.profiles_dir / f"{voice_id}.json").resolve()
        p.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def delete(self, voice_id: str) -> bool:
        p = (self.profiles_dir / f"{voice_id}.json").resolve()
        if p.exists():
            p.unlink()
            return True
        return False

    def resolve_path(self, rel_or_abs: str) -> Path:
        rp = Path(rel_or_abs)
        if rp.is_absolute():
            return rp
        return (self.voices_dir / rp).resolve()

    def export_pack(self, voice_id: str, out_zip: Path) -> Path:
        import zipfile
        p = self.get(voice_id)
        if not p:
             raise ValueError(f"Voice {voice_id} not found")

        with zipfile.ZipFile(out_zip, "w") as z:
            # 1. Profile JSON
            prof_file = self.profiles_dir / f"{voice_id}.json"
            z.write(prof_file, arcname=f"profiles/{voice_id}.json")

            # 2. Assets (if clone)
            if p.type == "clone" and p.clone:
                if p.clone.ref_audio_path:
                    audio_path = self.resolve_path(p.clone.ref_audio_path)
                    if audio_path.exists():
                        z.write(audio_path, arcname=p.clone.ref_audio_path)
                if p.clone.ref_text_path:
                    text_path = self.resolve_path(p.clone.ref_text_path)
                    if text_path.exists():
                        z.write(text_path, arcname=p.clone.ref_text_path)
        return out_zip

    def import_pack(self, zip_path: Path) -> str:
        import json
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            # Look for the profile JSON
            prof_entry = next((n for n in z.namelist() if n.startswith("profiles/") and n.endswith(".json")), None)
            if not prof_entry:
                raise ValueError("ZIP does not contain a profiles/*.json file")

            # Extract profile
            data = json.loads(z.read(prof_entry).decode("utf-8"))
            voice_id = data["id"]

            # Extract everything else
            for member in z.infolist():
                # Avoid ZipSlip: ensure path is within voices_dir
                target = (self.voices_dir / member.filename).resolve()
                if not str(target).startswith(str(self.voices_dir)):
                     continue
                z.extract(member, self.voices_dir)
            return voice_id
