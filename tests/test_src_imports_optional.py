from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "omnivoice_studio",  # expected top-level package name
        "omnivoice_studio.cli",  # expected CLI module
        "omnivoice_studio.config",  # expected config loader
        "omnivoice_studio.voices",  # expected voices manager
    ],
)
def test_src_modules_importable_if_present(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        pytest.skip(f"{module} not implemented yet")
