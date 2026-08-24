"""Test script for the OmniVoice Studio HTTP API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

BASE_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 30

log = logging.getLogger("test_api")


def test_list_voices():
    """Tests the /voices endpoint to list available voice profiles."""
    print("Listing voices...")
    resp = requests.get(f"{BASE_URL}/voices", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    print(resp.json())


def test_generate_custom():
    """Tests the /tts/custom_voice endpoint for custom voice generation."""
    print("Testing Custom Voice generation...")
    data = {
        "text": "Hello, this is a test of the custom voice system.",
        "speaker": "Vivian",
        "language": "English",
    }
    resp = requests.post(
        f"{BASE_URL}/tts/custom_voice", json=data, timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    res_data = resp.json()
    print("Success:", res_data)

    if res_data.get("audio_url"):
        audio_resp = requests.get(
            f"{BASE_URL}{res_data['audio_url']}", timeout=DEFAULT_TIMEOUT
        )
        audio_resp.raise_for_status()
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "test_custom.wav"
        out_file.write_bytes(audio_resp.content)
        print(f"Audio saved to: {out_file}")


def test_generate_design():
    """Tests the /tts/voice_design endpoint for voice design generation."""
    print("Testing Voice Design generation...")
    data = {
        "text": "I am a mysterious voice from the shadows.",
        "instruct": "A deep, gravelly male voice with a slight echo.",
        "language": "English",
    }
    resp = requests.post(
        f"{BASE_URL}/tts/voice_design", json=data, timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    res_data = resp.json()
    print("Success:", res_data)

    if res_data.get("audio_url"):
        audio_resp = requests.get(
            f"{BASE_URL}{res_data['audio_url']}", timeout=DEFAULT_TIMEOUT
        )
        audio_resp.raise_for_status()
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "test_design.wav"
        out_file.write_bytes(audio_resp.content)
        print(f"Audio saved to: {out_file}")


def test_generate_clone():
    """Tests the /tts/voice_clone endpoint."""
    print("Testing Voice Clone generation...")

    # We need a reference audio file
    wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
    if not wav_files:
        print("No wav files found in current directory for testing clone.")
        return

    ref_audio_path = os.path.abspath(wav_files[0])
    print(f"Using {ref_audio_path} as reference.")

    data = {
        "text": "This is a clone of the reference voice.",
        "ref_audio": ref_audio_path,
        "language": "English",
        "x_vector_only_mode": True,
    }
    resp = requests.post(
        f"{BASE_URL}/tts/voice_clone", json=data, timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    res_data = resp.json()
    print("Success:", res_data)

    if res_data.get("audio_url"):
        audio_resp = requests.get(
            f"{BASE_URL}{res_data['audio_url']}", timeout=DEFAULT_TIMEOUT
        )
        audio_resp.raise_for_status()
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "test_clone.wav"
        out_file.write_bytes(audio_resp.content)
        print(f"Audio saved to: {out_file}")


if __name__ == "__main__":
    # Ensure API is running before executing this
    try:
        test_list_voices()
        test_generate_custom()
        test_generate_design()
        test_generate_clone()
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it first.")
    except requests.RequestException as e:
        print(f"Test failed with request error: {e}")
    except RuntimeError as e:
        print(f"Test failed with runtime error: {e}")
