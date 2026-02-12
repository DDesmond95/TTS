from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class WarmupRequest(BaseModel):
    model: str


class RunResponse(BaseModel):
    run_id: str
    sample_rate: Optional[int] = None
    audio_url: Optional[str] = None
    run_dir: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class CustomVoiceRequest(BaseModel):
    text: Union[str, List[str]]
    language: Union[str, List[str]] = "Auto"
    speaker: Union[str, List[str]] = "Ryan"
    instruct: Union[str, List[str]] = ""
    model: Optional[str] = None
    gen: Dict[str, Any] = Field(default_factory=dict)


class VoiceDesignRequest(BaseModel):
    text: Union[str, List[str]]
    language: Union[str, List[str]] = "Auto"
    instruct: Union[str, List[str]] = ""
    model: Optional[str] = None
    gen: Dict[str, Any] = Field(default_factory=dict)


class VoiceCloneRequest(BaseModel):
    text: Union[str, List[str]]
    language: Union[str, List[str]] = "Auto"
    voice_profile: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    model: Optional[str] = None
    x_vector_only_mode: bool = False
    use_cached_prompt: bool = True
    gen: Dict[str, Any] = Field(default_factory=dict)


class DesignThenCloneRequest(BaseModel):
    design_text: str
    design_language: str = "Auto"
    design_instruct: str = ""
    clone_text: Union[str, List[str]]
    clone_language: Union[str, List[str]] = "Auto"
    voicedesign_model: Optional[str] = None
    base_model: Optional[str] = None
    gen_design: Dict[str, Any] = Field(default_factory=dict)
    gen_clone: Dict[str, Any] = Field(default_factory=dict)


class TokenizerEncodeRequest(BaseModel):
    audio: str
    model: Optional[str] = None


class TokenizerDecodeRequest(BaseModel):
    codes_json_path: str
    model: Optional[str] = None
