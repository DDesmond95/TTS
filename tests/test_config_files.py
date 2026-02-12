from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name", ["default.yaml", "dev.yaml", "prod.yaml", "docker.yaml"]
)
def test_config_exists(configs_dir: Path, name: str) -> None:
    assert (configs_dir / name).exists(), f"Missing configs/{name}"


def test_config_yaml_parses(configs_dir: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        pytest.skip("pyyaml not installed; install via project extras/dev deps")

    for p in sorted(configs_dir.glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{p} did not parse to a dict"
