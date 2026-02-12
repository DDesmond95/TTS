from __future__ import annotations

from pathlib import Path


def test_outputs_dir_exists(outputs_dir: Path) -> None:
    assert outputs_dir.exists()
    assert outputs_dir.is_dir()


def test_outputs_has_readme(outputs_dir: Path) -> None:
    readme = outputs_dir / "README.md"
    assert readme.exists(), "outputs/README.md is missing"
