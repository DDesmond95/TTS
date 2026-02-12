from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

VoiceType = Literal["customvoice", "clone", "design_template"]


class VoiceDefaults(BaseModel):
    language: str = "Auto"
    speaker: str = ""
    instruct: str = ""


class CloneConfig(BaseModel):
    ref_audio_path: str = ""
    ref_text_path: Optional[str] = ""
    x_vector_only_mode: bool = False
    cached_prompt_path: Optional[str] = ""


class DesignTemplateConfig(BaseModel):
    instruct_template: str = ""
    example_text: str = ""


class VoiceMeta(BaseModel):
    created_at: str
    updated_at: str


class VoiceProfile(BaseModel):
    id: str
    type: VoiceType
    display_name: str
    defaults: VoiceDefaults = Field(default_factory=VoiceDefaults)
    clone: Optional[CloneConfig] = None
    design_template: Optional[DesignTemplateConfig] = None
    meta: VoiceMeta
