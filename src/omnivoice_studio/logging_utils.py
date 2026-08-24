"""Logging utilities for OmniVoice Studio."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """
    Configures the global logging settings.

    Args:
        level: The logging level to use (e.g., 'DEBUG', 'INFO').
               If None, LOG_LEVEL env var or 'INFO' is used.
    """
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
