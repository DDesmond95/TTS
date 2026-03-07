import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ExampleSuite")

# Get project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from huggingface_hub import snapshot_download


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


OUTPUT_DIR = ROOT / "examples" / "output"
ensure_dir(OUTPUT_DIR)


# --- 1. OmniVoice Studio Engine ---
async def test_omnivoice():
    logger.info("Testing OmniVoice Studio Integrated Engine...")
    from omnivoice_studio.config import load_config
    from omnivoice_studio.engine.engine import TTSEngine

    # Use default config
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))

    engine = TTSEngine(
        models_dir=ROOT / cfg.paths.models_dir,
        voices_dir=ROOT / cfg.paths.voices_dir,
        outputs_dir=ROOT / cfg.paths.outputs_dir,
        runtime=cfg.runtime,
    )

    text = "This is a test of the OmniVoice Studio integrated engine."
    try:
        res = await engine.run_custom_voice(
            text=text, speaker="Ryan", language="English"
        )
        logger.info(f"OmniVoice Synthesis Success! Saved to: {res.audio_path}")
    except Exception as e:
        logger.error(f"OmniVoice Synthesis Failed: {e}")


# --- 2. Qwen3-TTS Direct Tasks ---
async def test_qwen3():
    logger.info("Testing Qwen3-TTS Tasks Directly...")
    from omnivoice_studio.config import load_config

    # We still need the engine for model loading/caching
    from omnivoice_studio.engine.engine import TTSEngine
    from omnivoice_studio.tasks.custom_voice import CustomVoiceRequest, CustomVoiceTask

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    engine = TTSEngine(
        ROOT / cfg.paths.models_dir,
        ROOT / cfg.paths.voices_dir,
        ROOT / cfg.paths.outputs_dir,
        cfg.runtime,
    )

    task = CustomVoiceTask()
    req = CustomVoiceRequest(text="Direct Qwen3 task execution test.")

    try:
        res = await task.run(engine, req)
        logger.info(f"Qwen3 Direct Task Success! Saved to: {res.audio_path}")
    except Exception as e:
        logger.error(f"Qwen3 Direct Task Failed: {e}")


# --- 3. MeanVC (Voice Conversion) ---
def test_meanvc():
    logger.info("Testing MeanVC Voice Conversion...")
    MEAN_VC_ROOT = ROOT / "MeanVC"
    if not MEAN_VC_ROOT.exists():
        logger.warning("MeanVC directory not found. Skipping.")
        return

    sys.path.append(str(MEAN_VC_ROOT))
    try:
        # Mocking or calling the inference script
        # Note: MeanVC requires specific .npy files for speaker embeddings
        logger.info("MeanVC requires pre-computed speaker embeddings (.npy).")
        logger.info("Run: python MeanVC/scripts/infer.py --help for standalone usage.")

        # In a real scenario, we would call:
        # from src.infer.infer import inference
        # but it has many dependencies.
        # We recommend using the provided scripts or the integrated API if available.
    except Exception as e:
        logger.error(f"MeanVC setup error: {e}")


# --- 4. TCSinger2 (Singing Synthesis) ---
def test_tcsinger():
    logger.info("Testing TCSinger2 Singing Synthesis...")
    TC_SINGER_ROOT = ROOT / "TCSinger2"
    if not TC_SINGER_ROOT.exists():
        logger.warning("TCSinger2 directory not found. Skipping.")
        return

    sys.path.append(str(TC_SINGER_ROOT))
    logger.info("TCSinger2 requires a trained checkpoint and manifest file.")
    logger.info("Run: python TCSinger2/scripts/test_sing.py --help for usage.")


# --- 5. VoiceSculptor (Voice Design) ---
def test_sculptor():
    logger.info("Testing VoiceSculptor...")
    SCULPTOR_ROOT = ROOT / "VoiceSculptor"
    if not SCULPTOR_ROOT.exists():
        logger.warning("VoiceSculptor directory not found. Skipping.")
        return

    sys.path.append(str(SCULPTOR_ROOT))
    # VoiceSculptor/infer.py can be called directly
    logger.info("VoiceSculptor can be run via: python VoiceSculptor/infer.py")


async def run_all():
    await test_omnivoice()
    await test_qwen3()
    test_meanvc()
    test_tcsinger()
    test_sculptor()


async def download_all_models():
    logger.info("Starting comprehensive model download...")

    # 1. Qwen3-TTS (Integrated)
    from omnivoice_studio.config import load_config

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    m_dir = ROOT / cfg.paths.models_dir
    m_dir.mkdir(parents=True, exist_ok=True)

    qwen_models = [
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ]
    for mid in qwen_models:
        target = m_dir / mid.split("/")[-1]
        logger.info(f"Downloading {mid} -> {target}")
        snapshot_download(
            repo_id=mid, local_dir=str(target), local_dir_use_symlinks=False
        )

    # 2. MeanVC
    m_meanvc = m_dir / "MeanVC"
    m_meanvc.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading MeanVC -> {m_meanvc}")
    snapshot_download(
        repo_id="ASLP-lab/MeanVC",
        allow_patterns=[
            "model_200ms.safetensors",
            "meanvc_200ms.pt",
            "fastu2++.pt",
            "vocos.pt",
        ],
        local_dir=str(m_meanvc),
        local_dir_use_symlinks=False,
    )

    # 3. VoiceSculptor
    scf_codec = m_dir / "xcodec2"
    scf_vd = m_dir / "VoiceSculptor-VD"
    scf_codec.mkdir(parents=True, exist_ok=True)
    scf_vd.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading VoiceSculptor Codec -> {scf_codec}")
    snapshot_download(
        repo_id="HKUSTAudio/xcodec2",
        local_dir=str(scf_codec),
        local_dir_use_symlinks=False,
    )

    logger.info(f"Downloading VoiceSculptor VD -> {scf_vd}")
    snapshot_download(
        repo_id="ASLP-lab/VoiceSculptor-VD",
        local_dir=str(scf_vd),
        local_dir_use_symlinks=False,
    )

    logger.info("All models downloaded successfully.")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Studio Example Suite")
    parser.add_argument(
        "--task",
        choices=[
            "omnivoice",
            "qwen3",
            "meanvc",
            "tcsinger",
            "sculptor",
            "all",
            "download",
        ],
        default="all",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    if args.task == "omnivoice":
        loop.run_until_complete(test_omnivoice())
    elif args.task == "qwen3":
        loop.run_until_complete(test_qwen3())
    elif args.task == "meanvc":
        test_meanvc()
    elif args.task == "tcsinger":
        test_tcsinger()
    elif args.task == "sculptor":
        test_sculptor()
    elif args.task == "download":
        loop.run_until_complete(download_all_models())
    else:
        loop.run_until_complete(run_all())


if __name__ == "__main__":
    main()
