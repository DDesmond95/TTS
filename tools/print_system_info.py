#!/usr/bin/env python3
import platform
import sys


def main() -> None:
    print("[system]")
    print(f"  python: {sys.version.splitlines()[0]}")
    print(f"  platform: {platform.platform()}")
    print(f"  executable: {sys.executable}")

    try:
        import torch

        print("\n[torch]")
        print(f"  torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            print(f"  device index: {idx}")
            print(f"  device name: {torch.cuda.get_device_name(idx)}")
            print(f"  capability: {torch.cuda.get_device_capability(idx)}")
            props = torch.cuda.get_device_properties(idx)
            print(f"  total vram: {props.total_memory / (1024**3):.2f} GB")
        else:
            print("  (cuda not available)")
    except Exception as e:
        print("\n[torch]")
        print(f"  torch not available or error: {e}")

    print("\n[recommended defaults]")
    print("  dtype: float16")
    print("  max_concurrent_jobs: 1")
    print("  model_cache_size: 1")
    print("  flash_attention: disabled")


if __name__ == "__main__":
    main()
