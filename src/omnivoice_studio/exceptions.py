"""Custom exception classes for OmniVoice Studio."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("omnivoice_studio.exceptions")


class OmniVoiceError(Exception):
    """Base exception for OmniVoice Studio"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(OmniVoiceError):
    """Raised when a model fails to load"""


class ModelNotFoundError(OmniVoiceError):
    """Raised when a requested model is not found in the registry"""


class InferenceError(OmniVoiceError):
    """Raised when an inference task fails"""


class AudioProcessingError(OmniVoiceError):
    """Raised when audio loading/processing fails"""


class ValidationError(OmniVoiceError):
    """Raised when request validation fails"""


class ConfigError(OmniVoiceError):
    """Raised when there is a configuration issue"""
