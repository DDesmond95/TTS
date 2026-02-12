from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "tts_platform",  # expected top-level package name
        "tts_platform.cli",  # expected CLI module
        "tts_platform.config",  # expected config loader
        "tts_platform.voices",  # expected voices manager
    ],
)
def test_src_modules_importable_if_present(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        pytest.skip(f"{module} not implemented yet")
