#!/usr/bin/env python3
import subprocess
import sys


def main() -> None:
    # This script expects to run on a host machine (not inside docker).
    # It checks:
    # - docker is installed
    # - nvidia-smi exists
    # - docker can see GPU with --gpus all
    print("[host] checking nvidia-smi")
    try:
        subprocess.check_call(["nvidia-smi"], stdout=sys.stdout, stderr=sys.stderr)
    except Exception as e:
        raise SystemExit(f"nvidia-smi not working: {e}")

    print("\n[host] checking docker")
    try:
        subprocess.check_call(
            ["docker", "version"], stdout=sys.stdout, stderr=sys.stderr
        )
    except Exception as e:
        raise SystemExit(f"docker not working: {e}")

    print("\n[host] checking docker --gpus all (requires NVIDIA Container Toolkit)")
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "nvidia/cuda:12.2.0-base-ubuntu22.04",
        "nvidia-smi",
    ]
    try:
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        print("\n[done] docker GPU access OK")
    except Exception as e:
        raise SystemExit(
            "docker GPU access failed. Ensure NVIDIA Container Toolkit is installed and your docker daemon is configured.\n"
            f"error: {e}"
        )


if __name__ == "__main__":
    main()
