"""Mixin for TTS task dispatching in OmniVoice Studio."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np

from ..pipelines.audiobook import AudiobookPipeline, AudiobookRequest
from ..pipelines.long_form import LongFormPipeline, LongFormRequest
from ..pipelines.npc_pack import NPCPackPipeline, NPCPackRequest
from ..pipelines.script_read import ScriptReadPipeline, ScriptReadRequest
from ..pipelines.subtitles import SubtitlesPipeline, SubtitlesRequest
from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
from ..tasks.design_then_clone import DesignThenCloneRequest, DesignThenCloneTask
from ..tasks.meanvc import VoiceConversionRequest, VoiceConversionTask
from ..tasks.tcsinger import SingingSynthesisRequest, SingingSynthesisTask
from ..tasks.tokenizer import (
    TokenizerDecodeRequest,
    TokenizerDecodeTask,
    TokenizerEncodeRequest,
    TokenizerEncodeTask,
)
from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask
from ..tasks.voice_design import VoiceDesignRequest, VoiceDesignTask
from ..tasks.voice_sculptor import VoiceSculptRequest, VoiceSculptTask

if TYPE_CHECKING:
    from pathlib import Path

    from ..storage.outputs import RunResult
    from .engine import TTSEngine


class TTSTaskRunnerMixin:
    """Mixin providing high-level run and stream methods for the TTSEngine."""

    async def run_custom_voice(
        self: TTSEngine,
        text: str | list[str],
        language: str | list[str] = "Auto",
        speaker: str | list[str] = "Ryan",
        instruct: str | list[str] = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a custom voice generation task.

        Args:
            text: Text to synthesize.
            language: Target language.
            speaker: Speaker name.
            instruct: Instruction text.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = CustomVoiceTask()
        req = CustomVoiceRequest(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            model=model,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_voice_design(
        self: TTSEngine,
        text: str | list[str],
        language: str | list[str] = "Auto",
        instruct: str | list[str] = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a voice design task.

        Args:
            text: Text to synthesize.
            language: Target language.
            instruct: Instruction text.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = VoiceDesignTask()
        req = VoiceDesignRequest(
            text=text,
            language=language,
            instruct=instruct,
            model=model,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_voice_clone(
        self: TTSEngine,
        text: str | list[str],
        language: str | list[str] = "Auto",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_profile: str | None = None,
        model: str | None = None,
        x_vector_only_mode: bool = False,
        use_cached_prompt: bool = True,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a voice cloning task.

        Args:
            text: Text to synthesize.
            language: Target language.
            ref_audio: Path to reference audio.
            ref_text: Text of the reference audio.
            voice_profile: Name of a saved voice profile.
            model: Model ID to use.
            x_vector_only_mode: Whether to use only x-vector.
            use_cached_prompt: Whether to use cached prompt.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = VoiceCloneTask()
        req = VoiceCloneRequest(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_profile=voice_profile,
            model=model,
            x_vector_only_mode=x_vector_only_mode,
            use_cached_prompt=use_cached_prompt,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_design_then_clone(
        self: TTSEngine,
        design_text: str,
        design_language: str,
        design_instruct: str,
        clone_text: str | list[str],
        clone_language: str | list[str] = "Auto",
        voicedesign_model: str | None = None,
        base_model: str | None = None,
        gen_design: dict[str, Any] | None = None,
        gen_clone: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Performs voice design and then clones that voice for a text synthesis.

        Args:
            design_text: Text for the design step.
            design_language: Language for design.
            design_instruct: Instruction for design.
            clone_text: Text for the cloning step.
            clone_language: Language for cloning.
            voicedesign_model: Model for design.
            base_model: Model for cloning.
            gen_design: Design generation parameters.
            gen_clone: Clone generation parameters.

        Returns:
            A RunResult object.
        """
        task = DesignThenCloneTask()
        req = DesignThenCloneRequest(
            design_text=design_text,
            design_language=design_language,
            design_instruct=design_instruct,
            clone_text=clone_text,
            clone_language=clone_language,
            voicedesign_model=voicedesign_model,
            base_model=base_model,
            gen_design=gen_design or {},
            gen_clone=gen_clone or {},
        )
        return await task.run(self, req)

    async def tokenizer_encode(
        self: TTSEngine, audio: str, model: str | None = None
    ) -> RunResult:
        """
        Encodes audio using a tokenizer model.

        Args:
            audio: Path to the audio file.
            model: Model ID to use.

        Returns:
            A RunResult object containing the codes.
        """
        task = TokenizerEncodeTask()
        req = TokenizerEncodeRequest(audio=audio, model=model)
        return await task.run(self, req)

    async def tokenizer_decode(
        self: TTSEngine, codes_json_path: str, model: str | None = None
    ) -> RunResult:
        """
        Decodes codes using a tokenizer model.

        Args:
            codes_json_path: Path to the JSON file with codes.
            model: Model ID to use.

        Returns:
            A RunResult object containing the audio.
        """
        task = TokenizerDecodeTask()
        req = TokenizerDecodeRequest(codes_json_path=codes_json_path, model=model)
        return await task.run(self, req)

    async def run_voice_conversion(
        self: TTSEngine,
        source_audio: str,
        target_speaker_audio: str | None = None,
        target_speaker_id: str | None = None,
        model: str | None = None,
        steps: int = 5,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a voice conversion task.

        Args:
            source_audio: Path to source audio.
            target_speaker_audio: Path to target speaker audio.
            target_speaker_id: ID of a target speaker from the registry.
            model: Model ID to use.
            steps: Number of diffusion steps.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = VoiceConversionTask()
        req = VoiceConversionRequest(
            source_audio=source_audio,
            target_speaker_audio=target_speaker_audio,
            target_speaker_id=target_speaker_id,
            model=model,
            steps=steps,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_singing_synthesis(
        self: TTSEngine,
        lyrics: str,
        score: str | None = None,
        ref_audio: str | None = None,
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a singing synthesis task.

        Args:
            lyrics: Lyrics text.
            score: Musical score info (JSON/text).
            ref_audio: Optional reference audio.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = SingingSynthesisTask()
        req = SingingSynthesisRequest(
            lyrics=lyrics,
            score=score,
            ref_audio=ref_audio,
            model=model,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_voice_sculpting(
        self: TTSEngine,
        instruction: str,
        ref_audio: str,
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a voice sculpting task (zero-shot editing).

        Args:
            instruction: Text instruction for the edit.
            ref_audio: Path to reference audio.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        task = VoiceSculptTask()
        req = VoiceSculptRequest(
            instruction=instruction,
            ref_audio=ref_audio,
            model=model,
            gen=gen or {},
        )
        return await task.run(self, req)

    async def run_long_form(
        self: TTSEngine,
        text: str,
        task_type: str = "custom_voice",
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a long-form synthesis pipeline.

        Args:
            text: Large text to synthesize.
            task_type: Type of task (e.g., "custom_voice").
            speaker: Speaker name.
            language: Target language.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object containing stitched audio.
        """
        pipe = LongFormPipeline()
        req = LongFormRequest(
            text=text,
            task_type=task_type,
            speaker=speaker,
            language=language,
            model=model,
            gen=gen or {},
        )
        return await pipe.run(self, req)

    async def run_npc_pack(
        self: TTSEngine,
        csv_path: str,
        speaker_map: dict[str, dict[str, Any]],
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a batch NPC lines generation pipeline.

        Args:
            csv_path: Path to CSV with NPC lines.
            speaker_map: Mapping of character names to voice configs.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object containing all generated files.
        """
        pipe = NPCPackPipeline()
        req = NPCPackRequest(
            csv_path=csv_path, speaker_map=speaker_map, model=model, gen=gen or {}
        )
        return await pipe.run(self, req)

    async def run_script_read(
        self: TTSEngine,
        script_text: str,
        speaker_map: dict[str, dict[str, Any]],
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        """
        Runs a script reading pipeline (multi-speaker dialogue).

        Args:
            script_text: Text of the script/screenplay.
            speaker_map: Mapping of speakers to voice configs.
            model: Model ID to use.
            gen: Generation parameters.

        Returns:
            A RunResult object.
        """
        pipe = ScriptReadPipeline()
        req = ScriptReadRequest(
            script_text=script_text,
            speaker_map=speaker_map,
            model=model,
            gen=gen or {},
        )
        return await pipe.run(self, req)

    async def run_audiobook(
        self: TTSEngine,
        chapter_paths: list[str],
        task_type: str = "custom_voice",
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
        merge_all: bool = True,
    ) -> RunResult:
        """
        Runs an audiobook generation pipeline.

        Args:
            chapter_paths: List of paths to chapter text files.
            task_type: Type of task to use.
            speaker: Speaker name.
            language: Target language.
            model: Model ID to use.
            gen: Generation parameters.
            merge_all: Whether to merge all chapters into one WAV.

        Returns:
            A RunResult object.
        """
        pipe = AudiobookPipeline()
        req = AudiobookRequest(
            chapter_paths=chapter_paths,
            task_type=task_type,
            speaker=speaker,
            language=language,
            model=model,
            gen=gen or {},
            merge_all=merge_all,
        )
        return await pipe.run(self, req)

    async def run_subtitles(
        self: TTSEngine,
        srt_path: str,
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
        preserve_timing: bool = True,
    ) -> RunResult:
        """
        Runs a subtitle dubbing pipeline.

        Args:
            srt_path: Path to the SRT file.
            speaker: Speaker name.
            language: Target language.
            model: Model ID to use.
            gen: Generation parameters.
            preserve_timing: Whether to match SRT timing (stretching).

        Returns:
            A RunResult object.
        """
        pipe = SubtitlesPipeline()
        req = SubtitlesRequest(
            srt_path=srt_path,
            speaker=speaker,
            language=language,
            model=model,
            gen=gen or {},
            preserve_timing=preserve_timing,
        )
        return await pipe.run(self, req)

    async def stream_custom_voice(
        self: TTSEngine,
        text: str,
        language: str = "Auto",
        speaker: str = "Ryan",
        instruct: str = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """
        Streams a custom voice generation task.

        Args:
            text: Text to synthesize.
            language: Target language.
            speaker: Speaker name.
            instruct: Instruction text.
            model: Model ID to use.
            gen: Generation parameters.

        Yields:
            Audio chunks as (waveform, sample_rate) tuples.
        """
        task = CustomVoiceTask()
        req = CustomVoiceRequest(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            model=model,
            gen=gen or {},
        )
        async for chunk in task.stream(self, req):
            yield chunk

    async def stream_voice_design(
        self: TTSEngine,
        text: str,
        language: str = "Auto",
        instruct: str = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """
        Streams a voice design task.

        Args:
            text: Text to synthesize.
            language: Target language.
            instruct: Instruction text.
            model: Model ID to use.
            gen: Generation parameters.

        Yields:
            Audio chunks as (waveform, sample_rate) tuples.
        """
        task = VoiceDesignTask()
        req = VoiceDesignRequest(
            text=text,
            language=language,
            instruct=instruct,
            model=model,
            gen=gen or {},
        )
        async for chunk in task.stream(self, req):
            yield chunk

    async def stream_voice_clone(
        self: TTSEngine,
        text: str,
        language: str = "Auto",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_profile: str | None = None,
        model: str | None = None,
        x_vector_only_mode: bool = False,
        use_cached_prompt: bool = True,
        gen: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """
        Streams a voice cloning task.

        Args:
            text: Text to synthesize.
            language: Target language.
            ref_audio: Path to reference audio.
            ref_text: Text of the reference audio.
            voice_profile: Name of a saved voice profile.
            model: Model ID to use.
            x_vector_only_mode: Whether to use only x-vector.
            use_cached_prompt: Whether to use cached prompt.
            gen: Generation parameters.

        Yields:
            Audio chunks as (waveform, sample_rate) tuples.
        """
        task = VoiceCloneTask()
        req = VoiceCloneRequest(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_profile=voice_profile,
            model=model,
            x_vector_only_mode=x_vector_only_mode,
            use_cached_prompt=use_cached_prompt,
            gen=gen or {},
        )
        async for chunk in task.stream(self, req):
            yield chunk

    def export_run(self: TTSEngine, run_id: str) -> Path:
        """
        Exports a run's results as a ZIP file.

        Args:
            run_id: The ID of the run to export.

        Returns:
            The path to the generated ZIP file.
        """
        _, run_dir = self.outputs.new_run_dir("export_run")
        zip_path = run_dir / f"{run_id}.zip"
        return self.outputs.export_run(run_id, zip_path)
