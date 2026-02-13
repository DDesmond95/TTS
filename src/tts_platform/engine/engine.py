from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RuntimeConfig
from ..models.registry import ModelRegistry
from ..storage.outputs import OutputManager, RunResult
from ..voices.store import VoiceStore

log = logging.getLogger("tts_platform.engine")


@dataclass
class Loaded:
    model_id: str
    kind: str  # base | customvoice | voicedesign | tokenizer
    obj: Any


class TTSEngine:
    """
    Single-GPU friendly engine:
    - max_concurrent_jobs enforced via semaphore (default 1)
    - simple LRU cache for loaded models (default 1)
    - no FlashAttention usage; attn_implementation=None
    """

    def __init__(
        self,
        models_dir: Path,
        voices_dir: Path,
        outputs_dir: Path,
        runtime: RuntimeConfig,
    ):
        self.registry = ModelRegistry(models_dir)
        self.voices = VoiceStore(voices_dir)
        self.outputs = OutputManager(outputs_dir)
        self.runtime = runtime

        self._sem = asyncio.Semaphore(max(1, int(runtime.max_concurrent_jobs)))
        self._cache: OrderedDict[str, Loaded] = OrderedDict()

    def list_models(self) -> list[dict]:
        out = []
        for m in self.registry.scan():
            out.append({"name": m.name, "kind": m.kind, "path": str(m.path)})
        return out

    def list_voices(self) -> list[dict]:
        out = []
        for v in self.voices.list_profiles():
            out.append({"id": v.id, "type": v.type, "display_name": v.display_name})
        return out

    def save_voice(self, voice_id: str, profile_data: dict) -> None:
        from ..voices.schema import VoiceProfile
        profile = VoiceProfile.model_validate(profile_data)
        self.voices.save(voice_id, profile)

    def delete_voice(self, voice_id: str) -> bool:
        return self.voices.delete(voice_id)

    def export_voice(self, voice_id: str) -> Path:
        run_id, run_dir = self.outputs.new_run_dir("export")
        zip_path = run_dir / f"{voice_id}.zip"
        return self.voices.export_pack(voice_id, zip_path)

    def import_voice(self, zip_path: str | Path) -> str:
        return self.voices.import_pack(Path(zip_path))

    def _device_dtype(self) -> tuple[str, Any]:
        import torch

        device = self.runtime.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        dtype = self.runtime.dtype.lower()
        if device.startswith("cuda"):
            torch_dtype = (
                torch.float16
                if dtype in ("float16", "fp16")
                else torch.bfloat16 if dtype in ("bfloat16", "bf16") else torch.float16
            )
        else:
            torch_dtype = torch.float32
        return device, torch_dtype

    def _infer_kind_from_id(self, model_id_or_path: str) -> str:
        n = model_id_or_path.lower()
        if "tokenizer" in n:
            return "tokenizer"
        if "voicedesign" in n:
            return "voicedesign"
        if "customvoice" in n:
            return "customvoice"
        if "base" in n:
            return "base"
        return "unknown"

    def _resolve_model(
        self, model: str | None, expected_kind: str | None = None
    ) -> str:
        """
        Resolve either:
        - explicit 'model' (HF id OR local path OR local folder name)
        - otherwise pick a sane default from models_dir (if present)
        """
        if model:
            # if it's a local folder name inside models_dir, use its path
            mi = self.registry.get(model)
            if mi:
                return str(mi.path)
            return model

        # auto-pick from local registry by expected kind
        scanned = self.registry.scan()
        if expected_kind:
            for m in scanned:
                if m.kind == expected_kind:
                    return str(m.path)
        # fallback: any model
        if scanned:
            return str(scanned[0].path)
        raise RuntimeError("No model provided and no local models found in models_dir")

    def _get_or_load(self, model_id_or_path: str, kind: str) -> Any:
        # LRU cache
        if model_id_or_path in self._cache:
            self._cache.move_to_end(model_id_or_path)
            return self._cache[model_id_or_path].obj

        device, torch_dtype = self._device_dtype()

        if kind == "tokenizer":
            from qwen_tts import Qwen3TTSTokenizer

            obj = Qwen3TTSTokenizer.from_pretrained(
                model_id_or_path,
                device_map=device,
            )
        else:
            from qwen_tts import Qwen3TTSModel

            obj = Qwen3TTSModel.from_pretrained(
                model_id_or_path,
                device_map=device,
                dtype=torch_dtype,
                attn_implementation=None,  # explicitly no FlashAttention
            )

        self._cache[model_id_or_path] = Loaded(
            model_id=model_id_or_path, kind=kind, obj=obj
        )
        self._cache.move_to_end(model_id_or_path)

        # evict
        while len(self._cache) > max(1, int(self.runtime.model_cache_size)):
            k, _ = self._cache.popitem(last=False)
            log.info("evicted model from cache: %s", k)

        return obj

    async def warmup(self, model: str) -> dict:
        model_id_or_path = self._resolve_model(model)
        kind = self._infer_kind_from_id(model_id_or_path)
        async with self._sem:
            _ = await asyncio.to_thread(self._get_or_load, model_id_or_path, kind)
        return {"ok": True, "model": model_id_or_path, "kind": kind}

    async def run_custom_voice(
        self,
        text: str | list[str],
        language: str | list[str] = "Auto",
        speaker: str | list[str] = "Ryan",
        instruct: str | list[str] = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
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
        self,
        text: str | list[str],
        language: str | list[str] = "Auto",
        instruct: str | list[str] = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        from ..tasks.voice_design import VoiceDesignRequest, VoiceDesignTask
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
        self,
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
        from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask
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
        self,
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
        from ..tasks.design_then_clone import DesignThenCloneRequest, DesignThenCloneTask
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
        self, audio: str, model: str | None = None
    ) -> RunResult:
        from ..tasks.tokenizer import TokenizerEncodeRequest, TokenizerEncodeTask
        task = TokenizerEncodeTask()
        req = TokenizerEncodeRequest(audio=audio, model=model)
        return await task.run(self, req)

    async def tokenizer_decode(
        self, codes_json_path: str, model: str | None = None
    ) -> RunResult:
        from ..tasks.tokenizer import TokenizerDecodeRequest, TokenizerDecodeTask
        task = TokenizerDecodeTask()
        req = TokenizerDecodeRequest(codes_json_path=codes_json_path, model=model)
        return await task.run(self, req)

    # --- Pipeline Runners ---

    async def run_long_form(
        self,
        text: str,
        task_type: str = "custom_voice",
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        from ..pipelines.long_form import LongFormPipeline, LongFormRequest
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
        self,
        csv_path: str,
        speaker_map: dict[str, dict[str, Any]],
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        from ..pipelines.npc_pack import NPCPackPipeline, NPCPackRequest
        pipe = NPCPackPipeline()
        req = NPCPackRequest(
            csv_path=csv_path, speaker_map=speaker_map, model=model, gen=gen or {}
        )
        return await pipe.run(self, req)

    async def run_script_read(
        self,
        script_text: str,
        speaker_map: dict[str, dict[str, Any]],
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> RunResult:
        from ..pipelines.script_read import ScriptReadPipeline, ScriptReadRequest
        pipe = ScriptReadPipeline()
        req = ScriptReadRequest(
            script_text=script_text,
            speaker_map=speaker_map,
            model=model,
            gen=gen or {},
        )
        return await pipe.run(self, req)

    async def run_audiobook(
        self,
        chapter_paths: list[str],
        task_type: str = "custom_voice",
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
        merge_all: bool = True,
    ) -> RunResult:
        from ..pipelines.audiobook import AudiobookPipeline, AudiobookRequest
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
        self,
        srt_path: str,
        speaker: str = "Ryan",
        language: str = "Auto",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
        preserve_timing: bool = True,
    ) -> RunResult:
        from ..pipelines.subtitles import SubtitlesPipeline, SubtitlesRequest
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
        self,
        text: str,
        language: str = "Auto",
        speaker: str = "Ryan",
        instruct: str = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        from ..tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask
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
        self,
        text: str,
        language: str = "Auto",
        instruct: str = "",
        model: str | None = None,
        gen: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        from ..tasks.voice_design import VoiceDesignRequest, VoiceDesignTask
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
        self,
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
        from ..tasks.voice_clone import VoiceCloneRequest, VoiceCloneTask
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

    def export_run(self, run_id: str) -> Path:
        _, run_dir = self.outputs.new_run_dir("export_run")
        zip_path = run_dir / f"{run_id}.zip"
        return self.outputs.export_run(run_id, zip_path)

    @staticmethod
    def wav_to_pcm16_bytes(wav: np.ndarray) -> bytes:
        x = np.asarray(wav, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        return i16.tobytes()
