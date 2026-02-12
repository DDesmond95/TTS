from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # Assume tests/ is directly under repo root.
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def voices_dir(repo_root: Path) -> Path:
    return Path(os.getenv("VOICES_DIR", repo_root / "voices")).resolve()


@pytest.fixture(scope="session")
def outputs_dir(repo_root: Path) -> Path:
    return Path(os.getenv("OUTPUTS_DIR", repo_root / "outputs")).resolve()


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return Path(os.getenv("CONFIGS_DIR", repo_root / "configs")).resolve()
