import os

import requests

BASE_URL = "http://localhost:8001"


def test_list_voices():
    print("Listing voices...")
    resp = requests.get(f"{BASE_URL}/voices")
    print(resp.json())


def test_generate_custom():
    print("Testing Custom Voice generation...")
    data = {
        "text": "Hello, this is a test of the custom voice system.",
        "speaker": "Vivian",
        "language": "English",
    }
    resp = requests.post(f"{BASE_URL}/generate/custom", json=data)
    if resp.status_code == 200:
        with open("outputs/test_custom.wav", "wb") as f:
            f.write(resp.content)
        print("Success: outputs/test_custom.wav saved.")
    else:
        print(f"Error: {resp.text}")


def test_generate_design():
    print("Testing Voice Design generation...")
    data = {
        "text": "I am a mysterious voice from the shadows.",
        "instruct": "A deep, gravelly male voice with a slight echo.",
        "language": "English",
    }
    resp = requests.post(f"{BASE_URL}/generate/design", json=data)
    if resp.status_code == 200:
        with open("outputs/test_design.wav", "wb") as f:
            f.write(resp.content)
        print("Success: outputs/test_design.wav saved.")
    else:
        print(f"Error: {resp.text}")


def test_save_raw_and_clone():
    print("Testing saving raw voice and cloning...")
    # Using a dummy file if needed, but assuming user has one
    # For now, we'll try to find any wav file in the dir
    wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
    if not wav_files:
        print("No wav files found for testing clone.")
        return

    ref_audio = wav_files[0]
    print(f"Using {ref_audio} as reference.")

    # Save as raw voice
    with open(ref_audio, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/voices/save_raw",
            data={"name": "test_voice"},
            files={"ref_audio": f},
        )
    print("Save raw response:", resp.json())

    # Clone it
    data = {
        "text": "This is a clone of the reference voice.",
        "voice_name": "test_voice",
        "language": "English",
        "use_xvector": "true",
    }
    resp = requests.post(f"{BASE_URL}/generate/clone", data=data)
    if resp.status_code == 200:
        with open("outputs/test_clone.wav", "wb") as f:
            f.write(resp.content)
        print("Success: outputs/test_clone.wav saved.")
    else:
        print(f"Error: {resp.text}")


if __name__ == "__main__":
    # Ensure API is running before executing this
    try:
        test_list_voices()
        test_generate_custom()
        test_generate_design()
        # test_save_raw_and_clone() # Uncomment if you have a wav file in the root
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it with run_api.cmd first.")
