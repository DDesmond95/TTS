import gc
import os

import torch
from qwen_tts import Qwen3TTSModel


class ModelWrapper:
    def __init__(self, models_root="d:/CodeAlpha/Projects/YTProjects/TTS/models"):
        self.models_root = models_root
        self.current_model = None
        self.current_model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name):
        if self.current_model_id == model_name:
            return self.current_model

        # Free memory
        if self.current_model:
            del self.current_model
            self.current_model = None
            torch.cuda.empty_cache()
            gc.collect()

        model_path = os.path.join(self.models_root, model_name)
        print(f"Loading model from {model_path}...")

        self.current_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation=None,  # Use manual to allow workaround
        )

        # Apply the user's workaround for batch processing
        try:
            print("Applying batch workaround (disabling sliding window)...")
            self.current_model.model.speech_tokenizer.model.decoder.pre_transformer.has_sliding_layers = (
                False
            )
            for (
                layer
            ) in (
                self.current_model.model.speech_tokenizer.model.decoder.pre_transformer.layers
            ):
                layer.attention_type = "full_attention"
        except AttributeError as e:
            print(
                f"Note: Could not apply workaround (might be a different model structure): {e}"
            )

        self.current_model_id = model_name
        return self.current_model

    def get_model_for_method(self, method):
        # Map method to the best available 1.7B model
        mapping = {
            "clone": "Qwen3-TTS-12Hz-1.7B-Base",
            "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        }
        return self.load_model(mapping.get(method))
