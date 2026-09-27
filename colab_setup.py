#!/usr/bin/env python3
"""Prepare an isolated CUDA-enabled Python environment on Google Colab."""

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv-colab"
PYTHON = ENVIRONMENT / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.txt"


def run(*command):
    command = [str(part) for part in command]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    if not Path("/content").exists():
        raise SystemExit(
            "This installer is intended for Google Colab. "
            "See README.md for local installation."
        )

    if not REQUIREMENTS.exists():
        raise SystemExit("requirements.txt was not found.")

    if shutil.which("nvidia-smi") is None:
        raise SystemExit(
            "No NVIDIA GPU was found.\n"
            "In Colab select Runtime > Change runtime type > T4 GPU,\n"
            "then reconnect and run this installer again."
        )

    run("nvidia-smi")

    run("apt-get", "update", "-qq")
    run(
        "apt-get",
        "install",
        "-y",
        "-qq",
        "espeak-ng",
        "libespeak-ng1",
        "libsndfile1",
        "build-essential",
        "cmake",
        "pkg-config",
    )

    run(sys.executable, "-m", "pip", "install", "--upgrade", "uv")
    uv = [sys.executable, "-m", "uv"]

    if not PYTHON.exists():
        run(
            *uv,
            "venv",
            "--seed",
            "--python",
            "3.12",
            ENVIRONMENT,
        )

    run(
        PYTHON,
        "-c",
        (
            "import sys; "
            "assert sys.version_info[:2] == (3, 12), sys.version; "
            "print('Environment Python:', sys.version)"
        ),
    )

    # Misaki/spaCy may invoke pip inside the environment.
    run(
        *uv,
        "pip",
        "install",
        "--python",
        PYTHON,
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    )

    # Explicit CUDA wheel. Reinstall to replace any accidental CPU build.
    run(
        *uv,
        "pip",
        "install",
        "--python",
        PYTHON,
        "--reinstall-package",
        "torch",
        "torch==2.8.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    )

    run(
        *uv,
        "pip",
        "install",
        "--python",
        PYTHON,
        "-r",
        REQUIREMENTS,
    )

    run(*uv, "pip", "check", "--python", PYTHON)

    # Validate imports and execute a real CUDA operation.
    check_code = """
import torch
import gradio
import kokoro
import misaki
import soundfile
from opencc import OpenCC
from deep_translator import GoogleTranslator

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Gradio:", gradio.__version__)
assert torch.cuda.is_available(), (
    "CUDA is unavailable in the application environment. "
    "Check that a GPU is allocated to this Colab session."
)

print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert y[0, 0].item() == 32.0
print("CUDA computation passed.")
"""
    run(PYTHON, "-c", check_code)

    # UniDic is needed only for Japanese.
    # Set INSTALL_JAPANESE=0 to defer its download until first Japanese use.
    if os.getenv("INSTALL_JAPANESE", "1") == "1":
        dictionary_check = subprocess.run(
            [
                str(PYTHON),
                "-c",
                (
                    "import pathlib, sys, unidic; "
                    "sys.exit(not "
                    "(pathlib.Path(unidic.DICDIR) / 'mecabrc').exists())"
                ),
            ],
            cwd=ROOT,
        )

        if dictionary_check.returncode != 0:
            run(PYTHON, "-m", "unidic", "download")

    print(
        "\nSetup finished.\n"
        "The model and voices will download when first used.\n\n"
        "Launch command:\n"
        f"{PYTHON} {ROOT / 'app.py'} --device cuda --share\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
