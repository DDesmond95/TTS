import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path("models")

MODEL_REPOS = {
    # Tokenizer (required for all models)
    "Qwen3-TTS-Tokenizer-12Hz": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    # CustomVoice models
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    # Base (voice clone)
    "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    # VoiceDesign
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


# -----------------------------
# Download helper
# -----------------------------
def download_model(repo_id: str, local_name: str):
    target_dir = BASE_DIR / local_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading: {repo_id} ===")
    print(f"→ Local path: {target_dir.resolve()}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # REQUIRED for Windows, safe on Linux
        resume_download=True,
    )

    print("✓ Done")


# -----------------------------
# Main
# -----------------------------
def main():
    print("Python version :", sys.version.split()[0])
    print("Platform       :", sys.platform)
    print("Base directory :", BASE_DIR.resolve())

    BASE_DIR.mkdir(exist_ok=True)

    for local_name, repo_id in MODEL_REPOS.items():
        download_model(repo_id, local_name)

    print("\nAll Qwen3-TTS models downloaded successfully.")


if __name__ == "__main__":
    main()
