"""Core orchestration engine for OmniVoice Studio TTS models."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

# MeanVC/VS/TCS path setup - must be before internal imports if they depend on it
_ENGINE_ROOT = Path(__file__).resolve().parents[3]

for subroot in ["MeanVC", "VoiceSculptor", "TCSinger2"]:
    prat = _ENGINE_ROOT / subroot
    if prat.exists() and str(prat) not in sys.path:
        sys.path.append(str(prat))

try:
    from vocos import Vocos
except ImportError:
    Vocos = Any  # type: ignore

try:
    from xcodec2.modeling_xcodec2 import XCodec2Model
except ImportError:
    XCodec2Model = Any  # type: ignore

try:
    from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = Any  # type: ignore
    CFMSampler = Any  # type: ignore
    instantiate_from_config = Any  # type: ignore

from ..config import RuntimeConfig
from ..exceptions import ModelLoadError, ModelNotFoundError
from ..models.registry import ModelRegistry
from ..storage.outputs import OutputStore, RunResult
from ..voices.schema import VoiceProfile
from ..voices.store import VoiceStore

try:
    # These are available after MeanVC root is in sys.path
    from src.infer.dit_kvcache import DiT
    from src.model.utils import load_checkpoint
except ImportError:
    DiT = Any  # type: ignore
    load_checkpoint = Any  # type: ignore
from .task_runner import TTSTaskRunnerMixin

log = logging.getLogger("omnivoice_studio.engine")


@dataclass
class Loaded:
    """Container for a loaded model and its metadata."""

    model_id: str
    kind: str  # base | customvoice | voicedesign | tokenizer
    obj: Any


class TTSEngine(TTSTaskRunnerMixin):
    """The orchestrator for all TTS tasks and model management."""

    def __init__(
        self,
        models_dir: Path,
        voices_dir: Path,
        outputs_dir: Path,
        runtime: RuntimeConfig,
    ):
        """Initializes the TTSEngine."""
        self.registry = ModelRegistry(models_dir)
        self.voices = VoiceStore(voices_dir)
        self.outputs = OutputStore(outputs_dir)
        self.runtime = runtime

        # device/dtype
        self.device = runtime.device
        self.torch_dtype = runtime.torch_dtype

        # Model cache
        self._cache: OrderedDict[str, Loaded] = OrderedDict()
        self._max_cache = runtime.model_cache_size

        # Simple semaphore for GPU concurrency
        self.sem = asyncio.Semaphore(max(1, int(runtime.max_concurrent_jobs)))

    def list_models(self) -> list[dict[str, Any]]:
        """Lists all discovered models and their metadata."""
        return [
            {"name": m.name, "path": str(m.path.resolve()), "kind": m.kind}
            for m in self.registry.discover()
        ]

    def list_voices(self) -> list[dict[str, Any]]:
        """Lists all available voice profiles."""
        return self.voices.list_all()

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Retrieves a voice profile by its ID."""
        return self.voices.get(voice_id)

    def save_voice(self, voice_id: str, profile: dict[str, Any]) -> None:
        """Saves or updates a voice profile."""
        vp = VoiceProfile(**profile)
        self.voices.save(voice_id, vp)

    def delete_voice(self, voice_id: str) -> bool:
        """Deletes a voice profile."""
        return self.voices.delete(voice_id)

    def _device_dtype(self) -> tuple[str, torch.dtype]:
        """Infers the best torch device and dtype from config."""
        device = self.runtime.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dt = self.runtime.dtype
        torch_dt = torch.float32
        if dt in ("float16", "fp16"):
            torch_dt = torch.float16
        elif dt in ("bfloat16", "bf16"):
            torch_dt = torch.bfloat16

        return device, torch_dt

    def infer_kind(self, model_id: str) -> str:
        """Infers the kind of a model given its ID or path."""
        return self.registry.infer_kind(model_id)

    def resolve_model(self, model: str | None, expected_kind: str | None = None) -> str:
        """
        Resolve a model reference to an absolute local path or HF ID.

        If model is None, it attempts to find a suitable default in the registry.

        Args:
            model: The model ID, path, or None to auto-pick.
            expected_kind: The expected kind of the model if auto-picking.

        Returns:
            The resolved model ID or absolute path.

        Raises:
            ModelNotFoundError: If no suitable model is found.
        """
        resolved: str | None = None

        if model:
            # 1. Check registry
            mi = self.registry.get(model)
            if mi:
                resolved = str(mi.path)
            else:
                # 2. Check if it's a direct valid path
                p = Path(model)
                if p.exists() and p.is_dir():
                    resolved = str(p.resolve())
                else:
                    # 3. Assume it's an HF ID (let transformers handle it later or fail)
                    resolved = model
        else:
            # auto-pick from local registry by expected kind
            scanned = self.registry.discover()
            if expected_kind:
                for m in scanned:
                    if m.kind == expected_kind:
                        resolved = str(m.path)
                        break

            # fallback: first available model if kind not found or not specified
            if not resolved and scanned:
                resolved = str(scanned[0].path)

        if not resolved:
            err_msg = (
                f"No model provided and no local models found for kind: {expected_kind}"
                if expected_kind
                else "No model provided and no local models found in registry"
            )
            raise ModelNotFoundError(err_msg)

        return resolved

    def get_or_load(self, model_id_or_path: str, kind: str) -> Any:
        """
        Get a model from cache or load it if missing.

        Args:
            model_id_or_path: The ID or path of the model to load.
            kind: The kind of the model (e.g., "base", "tokenizer").

        Returns:
            The loaded model object.

        Raises:
            ModelLoadError: If the model fails to load or the kind is unsupported.
        """
        # LRU cache
        if model_id_or_path in self._cache:
            self._cache.move_to_end(model_id_or_path)
            return self._cache[model_id_or_path].obj

        device, torch_dtype = self._device_dtype()

        try:
            obj = None
            if kind == "tokenizer":
                obj = self._load_tokenizer(model_id_or_path, device)
            elif kind in ("base", "customvoice", "voicedesign"):
                obj = self._load_qwen3(model_id_or_path, kind, device, torch_dtype)
            elif kind == "meanvc":
                obj = self._load_meanvc(model_id_or_path, device)
            elif kind == "voicesculptor":
                obj = self._load_voicesculptor(model_id_or_path, device, torch_dtype)
            elif kind == "xcodec2":
                obj = self._load_xcodec2(model_id_or_path, device)
            elif kind == "tcsinger":
                obj = self._load_tcsinger(model_id_or_path, device)
            else:
                raise ModelLoadError(f"Unsupported model kind: {kind}")

            # Evict if full
            if len(self._cache) >= self._max_cache:
                self._cache.popitem(last=False)

            self._cache[model_id_or_path] = Loaded(
                model_id=model_id_or_path, kind=kind, obj=obj
            )
            return obj

        except Exception as e:
            if not isinstance(e, ModelLoadError):
                log.exception("Failed to load model %s (%s)", model_id_or_path, kind)
                raise ModelLoadError(
                    f"Failed to load {kind} model {model_id_or_path}: {e}"
                ) from e
            raise

    def _load_tokenizer(self, model_id_or_path: str, device: str) -> Any:
        """
        Loads a Qwen3TTS tokenizer.

        Args:
            model_id_or_path: The ID or path of the tokenizer model.
            device: The device to load the tokenizer on.

        Returns:
            The loaded tokenizer object.
        """
        obj = Qwen3TTSTokenizer.from_pretrained(model_id_or_path, device_map=device)
        return self._cache_and_return(model_id_or_path, "tokenizer", obj)

    def _load_qwen3(
        self, model_id_or_path: str, kind: str, device: str, torch_dtype: Any
    ) -> Any:
        """
        Loads a Qwen3TTS model (base, customvoice, or voicedesign).

        Args:
            model_id_or_path: The ID or path of the Qwen3TTS model.
            kind: The kind of the Qwen3TTS model.
            device: The device to load the model on.
            torch_dtype: The torch data type to use for the model.

        Returns:
            The loaded Qwen3TTS model object.
        """
        try:
            AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
            AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
            AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
        except Exception:
            pass

        cfg_obj = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
        if self.runtime.disable_sliding_window:
            cfg_obj.sliding_window = None

        obj = Qwen3TTSModel.from_pretrained(
            model_id_or_path,
            config=cfg_obj,
            device_map=device,
            dtype=torch_dtype,
            attn_implementation=self.runtime.attn_implementation or "eager",
        )
        return self._cache_and_return(model_id_or_path, kind, obj)

    def _load_meanvc(self, model_id_or_path: str, device: str) -> dict[str, Any]:
        """
        Loads a MeanVC model.

        Args:
            model_id_or_path: The ID or path of the MeanVC model.
            device: The device to load the model on.

        Returns:
            A dictionary containing the loaded MeanVC model and vocoder.

        Raises:
            ModelLoadError: If MeanVC components or files are missing.
        """
        if DiT is Any or load_checkpoint is Any:
            raise ModelLoadError("MeanVC model components not found in sys.path")

        ckpt_path = Path(model_id_or_path) / "model_200ms.safetensors"
        if not ckpt_path.exists():
            raise ModelLoadError(f"MeanVC checkpoint not found: {ckpt_path}")

        model = DiT(chunk_size=200).to(device)
        load_checkpoint(model, str(ckpt_path), device)
        model.eval()

        vocos_cfg = Path(model_id_or_path) / "config.yaml"
        vocos_pt = Path(model_id_or_path) / "vocos.pt"
        if not vocos_cfg.exists() or not vocos_pt.exists():
            raise ModelLoadError(f"MeanVC Vocos files missing in {model_id_or_path}")

        vocos = Vocos.from_hparams(str(vocos_cfg))
        vocos.load_state_dict(torch.load(str(vocos_pt), map_location=device))
        vocos.eval()

        obj = {"model": model, "vocos": vocos}
        return self._cache_and_return(model_id_or_path, "meanvc", obj)

    def _load_voicesculptor(
        self, model_id_or_path: str, device: str, torch_dtype: Any
    ) -> dict[str, Any]:
        """
        Loads a VoiceSculptor model.

        Args:
            model_id_or_path: The ID or path of the VoiceSculptor model.
            device: The device to load the model on.
            torch_dtype: The torch data type to use for the model.

        Returns:
            A dictionary containing the loaded VoiceSculptor model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        obj = {"model": model, "tokenizer": tokenizer}
        return self._cache_and_return(model_id_or_path, "voicesculptor", obj)

    def _load_xcodec2(self, model_id_or_path: str, device: str) -> Any:
        """
        Loads an XCodec2 model.

        Args:
            model_id_or_path: The ID or path of the XCodec2 model.
            device: The device to load the model on.

        Returns:
            The loaded XCodec2 model object.

        Raises:
            ModelLoadError: If XCodec2 components are missing.
        """
        if XCodec2Model is Any:
            raise ModelLoadError("XCodec2Model components not found in sys.path")

        obj = XCodec2Model.from_pretrained(model_id_or_path)
        obj.to(device)
        obj.eval()
        return self._cache_and_return(model_id_or_path, "xcodec2", obj)

    def _load_tcsinger(self, model_id_or_path: str, device: str) -> Any:
        """
        Loads a TCSinger model.

        Args:
            model_id_or_path: The ID or path of the TCSinger model.
            device: The device to load the model on.

        Returns:
            The loaded TCSinger model sampler object.

        Raises:
            ModelLoadError: If TCSinger components or files are missing.
        """
        if OmegaConf is Any or CFMSampler is Any or instantiate_from_config is Any:
            raise ModelLoadError(
                "TCSinger2 components (ldm/omegaconf) not found in sys.path"
            )

        try:
            from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
            from ldm.util import instantiate_from_config
        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import TCSinger2 components from {root}: {e}"
            ) from e

        config_path = Path(model_id_or_path) / "config.yaml"
        ckpt_path = Path(model_id_or_path) / "checkpoints" / "last.ckpt"
        if not config_path.exists() or not ckpt_path.exists():
            raise ModelLoadError(
                f"TCSinger2 config or checkpoint missing in {model_id_or_path}"
            )

        config = OmegaConf.load(str(config_path))
        model = instantiate_from_config(config.model)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location="cpu")["state_dict"],
            strict=False,
        )
        model = model.to(device)
        obj = CFMSampler(model, num_timesteps=1000)
        return self._cache_and_return(model_id_or_path, "tcsinger", obj)

    def _cache_and_return(self, model_id: str, kind: str, obj: Any) -> Any:
        """
        Caches a loaded model and returns it, applying LRU eviction if necessary.

        Args:
            model_id: The ID of the model to cache.
            kind: The kind of the model.
            obj: The loaded model object.

        Returns:
            The loaded model object.
        """
        self._cache[model_id] = Loaded(model_id=model_id, kind=kind, obj=obj)
        self._cache.move_to_end(model_id)

        while len(self._cache) > max(1, int(self.runtime.model_cache_size)):
            k, _ = self._cache.popitem(last=False)
            log.info("evicted model from cache: %s", k)
        return obj

    async def warmup(self, model: str) -> dict:
        """
        Force load a model into memory.

        Args:
            model: The model ID or path to warm up.

        Returns:
            A dictionary indicating the status, model ID, and kind of the warmed-up model.
        """
        model_id_or_path = self.resolve_model(model)
        kind = self.registry.infer_kind(Path(model_id_or_path).name)
        async with self.sem:
            # get_or_load might be CPU/GPU bound, but let's assume it's okay to call in thread
            await asyncio.to_thread(self.get_or_load, model_id_or_path, kind)
        return {"status": "ok", "model": model_id_or_path, "kind": kind}

    # Task and Pipeline methods are provided by TTSTaskRunnerMixin

    @staticmethod
    def wav_to_pcm16_bytes(wav: np.ndarray) -> bytes:
        """
        Converts a float32 waveform to 16-bit PCM bytes.

        Args:
            wav: Input float32 waveform array.

        Returns:
            PCM16 encoded bytes.
        """
        x = np.asarray(wav, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        return i16.tobytes()
