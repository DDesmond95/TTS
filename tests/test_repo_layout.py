from __future__ import annotations

from pathlib import Path


def test_expected_folders_exist(repo_root: Path) -> None:
    required = [
        "src",
        "docs",
        "outputs",
        "voices",
        "models",
        "tools",
        "configs",
        "tests",
        "docker",
        ".github",
        ".gitlab",
    ]
    for name in required:
        assert (repo_root / name).exists(), f"Missing folder: {name}"
