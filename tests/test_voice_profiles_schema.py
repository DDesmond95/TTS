from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALLOWED_TYPES = {"customvoice", "clone", "design_template"}


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_common(d: Dict[str, Any], p: Path) -> None:
    assert isinstance(d.get("id"), str) and d["id"], f"{p}: missing/invalid id"
    assert (
        isinstance(d.get("type"), str) and d["type"] in ALLOWED_TYPES
    ), f"{p}: invalid type"
    assert (
        isinstance(d.get("display_name"), str) and d["display_name"]
    ), f"{p}: missing display_name"
    defaults = d.get("defaults")
    assert isinstance(defaults, dict), f"{p}: defaults must be an object"
    assert (
        isinstance(defaults.get("language"), str) and defaults["language"]
    ), f"{p}: defaults.language missing"
    assert isinstance(
        defaults.get("instruct"), str
    ), f"{p}: defaults.instruct must be string"
    meta = d.get("meta")
    assert isinstance(meta, dict), f"{p}: meta must be an object"
    assert (
        isinstance(meta.get("created_at"), str) and meta["created_at"]
    ), f"{p}: meta.created_at missing"
    assert (
        isinstance(meta.get("updated_at"), str) and meta["updated_at"]
    ), f"{p}: meta.updated_at missing"


def _validate_customvoice(d: Dict[str, Any], p: Path) -> None:
    defaults = d["defaults"]
    assert (
        isinstance(defaults.get("speaker"), str) and defaults["speaker"]
    ), f"{p}: defaults.speaker missing"
    assert "clone" not in d, f"{p}: customvoice must not contain clone section"
    assert (
        "design_template" not in d
    ), f"{p}: customvoice must not contain design_template section"


def _validate_clone(d: Dict[str, Any], p: Path) -> None:
    clone = d.get("clone")
    assert isinstance(clone, dict), f"{p}: clone section required for clone type"
    assert (
        isinstance(clone.get("ref_audio_path"), str) and clone["ref_audio_path"]
    ), f"{p}: clone.ref_audio_path missing"
    # ref_text_path can be empty only if x_vector_only_mode is true
    xvec = bool(clone.get("x_vector_only_mode", False))
    ref_text = clone.get("ref_text_path")
    if xvec:
        assert ref_text is None or isinstance(
            ref_text, str
        ), f"{p}: clone.ref_text_path must be string or null"
    else:
        assert (
            isinstance(ref_text, str) and ref_text
        ), f"{p}: clone.ref_text_path required unless x_vector_only_mode=true"
    cached = clone.get("cached_prompt_path")
    assert cached is None or isinstance(
        cached, str
    ), f"{p}: clone.cached_prompt_path must be string or null"
    assert (
        "design_template" not in d
    ), f"{p}: clone must not contain design_template section"


def _validate_design_template(d: Dict[str, Any], p: Path) -> None:
    dt = d.get("design_template")
    assert isinstance(
        dt, dict
    ), f"{p}: design_template section required for design_template type"
    assert (
        isinstance(dt.get("instruct_template"), str) and dt["instruct_template"]
    ), f"{p}: instruct_template missing"
    assert (
        isinstance(dt.get("example_text"), str) and dt["example_text"]
    ), f"{p}: example_text missing"
    assert "clone" not in d, f"{p}: design_template must not contain clone section"


def test_voice_profiles_valid_json(voices_dir: Path) -> None:
    profiles_dir = voices_dir / "profiles"
    assert profiles_dir.exists(), "voices/profiles does not exist"
    files = sorted(profiles_dir.glob("*.json"))
    # It’s okay if empty early on, but usually you'll have at least a few.
    for p in files:
        data = _load_json(p)
        _validate_common(data, p)
        t = data["type"]
        if t == "customvoice":
            _validate_customvoice(data, p)
        elif t == "clone":
            _validate_clone(data, p)
        elif t == "design_template":
            _validate_design_template(data, p)


def test_clone_assets_exist_when_present(voices_dir: Path) -> None:
    profiles_dir = voices_dir / "profiles"
    for p in sorted(profiles_dir.glob("*.json")):
        d = _load_json(p)
        if d.get("type") != "clone":
            continue
        clone = d.get("clone") or {}
        ref_audio = clone.get("ref_audio_path")
        ref_text = clone.get("ref_text_path")
        if isinstance(ref_audio, str) and ref_audio:
            assert (
                voices_dir / ref_audio
            ).exists(), f"{p}: missing ref audio {ref_audio}"
        xvec = bool(clone.get("x_vector_only_mode", False))
        if not xvec and isinstance(ref_text, str) and ref_text:
            assert (voices_dir / ref_text).exists(), f"{p}: missing ref text {ref_text}"
