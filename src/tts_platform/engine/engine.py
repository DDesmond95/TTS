from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import RuntimeConfig
from ..models.registry import ModelRegistry
from ..voices.store import VoiceStore
from .outputs import OutputManager, RunResult

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
        model_id_or_path = self._resolve_model(model, expected_kind="customvoice")
        params = {
            "task": "custom_voice",
            "model": model_id_or_path,
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
            "gen": gen or {},
        }
        run_id, run_dir = self.outputs.new_run_dir("custom_voice")
        self.outputs.write_params(run_dir, params)

        async with self._sem:
            model_obj = await asyncio.to_thread(
                self._get_or_load, model_id_or_path, "customvoice"
            )
            wavs, sr = await asyncio.to_thread(
                model_obj.generate_custom_voice,
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                **(gen or {}),
            )

        # qwen-tts returns list of wav arrays
        audio_paths = []
        for i, w in enumerate(wavs):
            audio_paths.append(
                self.outputs.save_wav(
                    run_dir, np.asarray(w), int(sr), filename=f"audio_{i}.wav"
                )
            )

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        self.outputs.write_meta(run_dir, meta)

        # convenience: if single, also write audio.wav
        audio_path = None
        if len(audio_paths) == 1:
            audio_path = audio_paths[0]
            # alias
            if audio_path.name != "audio.wav":
                (run_dir / "audio.wav").write_bytes(audio_path.read_bytes())
                audio_path = (run_dir / "audio.wav").resolve()

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    async def run_voice_design(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Auto",
        instruct: Union[str, List[str]] = "",
        model: Optional[str] = None,
        gen: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        model_id_or_path = self._resolve_model(model, expected_kind="voicedesign")
        params = {
            "task": "voice_design",
            "model": model_id_or_path,
            "text": text,
            "language": language,
            "instruct": instruct,
            "gen": gen or {},
        }
        run_id, run_dir = self.outputs.new_run_dir("voice_design")
        self.outputs.write_params(run_dir, params)

        async with self._sem:
            model_obj = await asyncio.to_thread(
                self._get_or_load, model_id_or_path, "voicedesign"
            )
            wavs, sr = await asyncio.to_thread(
                model_obj.generate_voice_design,
                text=text,
                language=language,
                instruct=instruct,
                **(gen or {}),
            )

        audio_paths = []
        for i, w in enumerate(wavs):
            audio_paths.append(
                self.outputs.save_wav(
                    run_dir, np.asarray(w), int(sr), filename=f"audio_{i}.wav"
                )
            )

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        self.outputs.write_meta(run_dir, meta)

        audio_path = None
        if len(audio_paths) == 1:
            audio_path = audio_paths[0]
            if audio_path.name != "audio.wav":
                (run_dir / "audio.wav").write_bytes(audio_path.read_bytes())
                audio_path = (run_dir / "audio.wav").resolve()

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    async def run_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Auto",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_profile: Optional[str] = None,
        model: Optional[str] = None,
        x_vector_only_mode: bool = False,
        use_cached_prompt: bool = True,
        gen: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        model_id_or_path = self._resolve_model(model, expected_kind="base")
        run_id, run_dir = self.outputs.new_run_dir("voice_clone")

        # resolve via voice profile if provided
        cached_prompt = None
        if voice_profile:
            prof = self.voices.get(voice_profile)
            if not prof:
                raise ValueError(f"Voice profile not found: {voice_profile}")
            if prof.type != "clone" or not prof.clone:
                raise ValueError(
                    f"Voice profile is not a clone profile: {voice_profile}"
                )

            if not ref_audio:
                ref_audio = prof.clone.ref_audio_path
            if not ref_text and prof.clone.ref_text_path:
                ref_text = prof.clone.ref_text_path
            x_vector_only_mode = (
                bool(prof.clone.x_vector_only_mode)
                if prof.clone
                else x_vector_only_mode
            )

            if use_cached_prompt and prof.clone and prof.clone.cached_prompt_path:
                import torch

                prompt_path = self.voices.resolve_path(prof.clone.cached_prompt_path)
                if prompt_path.exists():
                    cached_prompt = torch.load(str(prompt_path), map_location="cpu")

        # resolve actual paths if they are voice-relative
        ref_audio_resolved = (
            self.voices.resolve_path(ref_audio).as_posix() if ref_audio else None
        )
        ref_text_str = None
        if ref_text and not x_vector_only_mode:
            rt = self.voices.resolve_path(ref_text)
            ref_text_str = rt.read_text(encoding="utf-8").strip()

        params = {
            "task": "voice_clone",
            "model": model_id_or_path,
            "text": text,
            "language": language,
            "voice_profile": voice_profile,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            "use_cached_prompt": use_cached_prompt,
            "gen": gen or {},
        }
        self.outputs.write_params(run_dir, params)

        async with self._sem:
            model_obj = await asyncio.to_thread(
                self._get_or_load, model_id_or_path, "base"
            )

            if cached_prompt is not None:
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_clone,
                    text=text,
                    language=language,
                    voice_clone_prompt=cached_prompt,
                    **(gen or {}),
                )
            else:
                if not ref_audio_resolved:
                    raise ValueError(
                        "ref_audio is required (or set voice_profile with ref_audio_path)"
                    )
                wavs, sr = await asyncio.to_thread(
                    model_obj.generate_voice_clone,
                    text=text,
                    language=language,
                    ref_audio=ref_audio_resolved,
                    ref_text=ref_text_str,
                    x_vector_only_mode=x_vector_only_mode,
                    **(gen or {}),
                )

        audio_paths = []
        for i, w in enumerate(wavs):
            audio_paths.append(
                self.outputs.save_wav(
                    run_dir, np.asarray(w), int(sr), filename=f"audio_{i}.wav"
                )
            )

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        self.outputs.write_meta(run_dir, meta)

        audio_path = None
        if len(audio_paths) == 1:
            audio_path = audio_paths[0]
            if audio_path.name != "audio.wav":
                (run_dir / "audio.wav").write_bytes(audio_path.read_bytes())
                audio_path = (run_dir / "audio.wav").resolve()

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    async def run_design_then_clone(
        self,
        design_text: str,
        design_language: str,
        design_instruct: str,
        clone_text: Union[str, List[str]],
        clone_language: Union[str, List[str]] = "Auto",
        voicedesign_model: Optional[str] = None,
        base_model: Optional[str] = None,
        gen_design: Optional[Dict[str, Any]] = None,
        gen_clone: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        # step 1: design a reference
        design_res = await self.run_voice_design(
            text=design_text,
            language=design_language,
            instruct=design_instruct,
            model=voicedesign_model,
            gen=gen_design,
        )

        # step 2: build clone prompt (in-memory) and then clone
        import torch
        from qwen_tts import Qwen3TTSModel

        base_id_or_path = self._resolve_model(base_model, expected_kind="base")
        device, torch_dtype = self._device_dtype()

        async with self._sem:
            base_obj = await asyncio.to_thread(
                self._get_or_load, base_id_or_path, "base"
            )

            # load the design wav (single file expectation)
            if not design_res.audio_path:
                raise RuntimeError("design step did not produce audio.wav")

            prompt_items = await asyncio.to_thread(
                base_obj.create_voice_clone_prompt,
                ref_audio=str(design_res.audio_path),
                ref_text=design_text,
                x_vector_only_mode=False,
            )

            # clone using prompt
            wavs, sr = await asyncio.to_thread(
                base_obj.generate_voice_clone,
                text=clone_text,
                language=clone_language,
                voice_clone_prompt=prompt_items,
                **(gen_clone or {}),
            )

        # store under a new run
        run_id, run_dir = self.outputs.new_run_dir("design_then_clone")
        self.outputs.write_params(
            run_dir,
            {
                "task": "design_then_clone",
                "voicedesign_run_id": design_res.run_id,
                "voicedesign_model": voicedesign_model,
                "base_model": base_id_or_path,
                "design_text": design_text,
                "design_language": design_language,
                "design_instruct": design_instruct,
                "clone_text": clone_text,
                "clone_language": clone_language,
                "gen_design": gen_design or {},
                "gen_clone": gen_clone or {},
            },
        )

        audio_paths = []
        for i, w in enumerate(wavs):
            audio_paths.append(
                self.outputs.save_wav(
                    run_dir, np.asarray(w), int(sr), filename=f"audio_{i}.wav"
                )
            )

        meta = {
            "sample_rate": int(sr),
            "count": len(audio_paths),
            "files": [p.name for p in audio_paths],
        }
        self.outputs.write_meta(run_dir, meta)

        audio_path = None
        if len(audio_paths) == 1:
            audio_path = audio_paths[0]
            if audio_path.name != "audio.wav":
                (run_dir / "audio.wav").write_bytes(audio_path.read_bytes())
                audio_path = (run_dir / "audio.wav").resolve()

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    async def tokenizer_encode(
        self, audio: str, model: Optional[str] = None
    ) -> RunResult:
        model_id_or_path = self._resolve_model(model, expected_kind="tokenizer")
        run_id, run_dir = self.outputs.new_run_dir("tokenizer_encode")
        params = {"task": "tokenizer_encode", "model": model_id_or_path, "audio": audio}
        self.outputs.write_params(run_dir, params)

        async with self._sem:
            tok = await asyncio.to_thread(
                self._get_or_load, model_id_or_path, "tokenizer"
            )
            enc = await asyncio.to_thread(tok.encode, audio)

        # store encoded payload
        out = (run_dir / "codes.json").resolve()
        import json

        out.write_text(json.dumps(enc, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = {"codes_path": out.name}
        self.outputs.write_meta(run_dir, meta)
        return RunResult(
            run_id=run_id, run_dir=run_dir, audio_path=None, sample_rate=None, meta=meta
        )

    async def tokenizer_decode(
        self, codes_json_path: str, model: Optional[str] = None
    ) -> RunResult:
        model_id_or_path = self._resolve_model(model, expected_kind="tokenizer")
        run_id, run_dir = self.outputs.new_run_dir("tokenizer_decode")
        params = {
            "task": "tokenizer_decode",
            "model": model_id_or_path,
            "codes_json_path": codes_json_path,
        }
        self.outputs.write_params(run_dir, params)

        codes_path = Path(codes_json_path).resolve()
        import json

        enc = json.loads(codes_path.read_text(encoding="utf-8"))

        async with self._sem:
            tok = await asyncio.to_thread(
                self._get_or_load, model_id_or_path, "tokenizer"
            )
            wavs, sr = await asyncio.to_thread(tok.decode, enc)

        audio_path = self.outputs.save_wav(
            run_dir, np.asarray(wavs[0]), int(sr), filename="audio.wav"
        )
        meta = {"sample_rate": int(sr), "files": [audio_path.name]}
        self.outputs.write_meta(run_dir, meta)
        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            audio_path=audio_path,
            sample_rate=int(sr),
            meta=meta,
        )

    @staticmethod
    def wav_to_pcm16_bytes(wav: np.ndarray) -> bytes:
        x = np.asarray(wav, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        return i16.tobytes()
