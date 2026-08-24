"""Pydantic schemas for OmniVoice Studio API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..pipelines.audiobook import AudiobookRequest
from ..pipelines.long_form import LongFormRequest
from ..pipelines.npc_pack import NPCPackRequest
from ..pipelines.script_read import ScriptReadRequest
from ..pipelines.subtitles import SubtitlesRequest
from ..tasks.custom_voice import CustomVoiceRequest
from ..tasks.design_then_clone import DesignThenCloneRequest
from ..tasks.meanvc import VoiceConversionRequest
from ..tasks.tcsinger import SingingSynthesisRequest
from ..tasks.tokenizer import TokenizerDecodeRequest, TokenizerEncodeRequest
from ..tasks.voice_clone import VoiceCloneRequest
from ..tasks.voice_design import VoiceDesignRequest
from ..tasks.voice_sculptor import VoiceSculptRequest
from ..voices.schema import VoiceProfile


class WarmupRequest(BaseModel):
    """Request for warming up a specific model."""

    model: str


class RunResponse(BaseModel):
    """Standard response for a successful task run."""

    run_id: str
    sample_rate: int | None = None
    audio_url: str | None = None
    run_dir: str
    meta: dict[str, Any] = Field(default_factory=dict)


# Re-exporting
__all__ = [
    "WarmupRequest",
    "RunResponse",
    "CustomVoiceRequest",
    "VoiceDesignRequest",
    "VoiceCloneRequest",
    "DesignThenCloneRequest",
    "VoiceConversionRequest",
    "SingingSynthesisRequest",
    "VoiceSculptRequest",
    "TokenizerEncodeRequest",
    "TokenizerDecodeRequest",
    "LongFormRequest",
    "NPCPackRequest",
    "ScriptReadRequest",
    "SubtitlesRequest",
    "AudiobookRequest",
    "VoiceProfile",
]
