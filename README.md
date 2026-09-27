# 🎙️ Kokoro Multilingual TTS + Translation

多語言語音合成＋翻譯，支援 Google Colab GPU 與本機執行。

A bilingual Gradio application for:

- Source text-to-speech
- Automatic language detection
- Optional translation and translated speech
- Editing and replaying translations
- Dialogue with one turn per line
- Manual language and voice selection
- Adjustable speech speed
- WAV audio playback and download
- CUDA acceleration when available

## Supported speech languages

| Language | Example voices |
|---|---|
| American English | af_heart, am_michael |
| British English | bf_emma, bm_george |
| Traditional Chinese, spoken as Mandarin | zf_xiaobei, zm_yunxi |
| Simplified Chinese, spoken as Mandarin | zf_xiaoxiao, zm_yunjian |
| Japanese | jf_alpha, jm_kumo |
| Spanish | ef_dora, em_alex |
| French | ff_siwis |
| Italian | if_sara, im_nicola |
| Brazilian Portuguese | pf_dora, pm_alex |
| Hindi | hf_alpha, hm_omega |

Traditional and Simplified Chinese are writing variants, not separate
spoken-language models.

Unsupported source languages use an English voice as a fallback.
This does not provide native pronunciation for unsupported languages.
Translation can still work when the translation service supports the
source language.

## Project files

```text
.
├── app.py
├── requirements.txt
├── colab_setup.py
├── README.md
├── .gitignore
└── LICENSE
```

## Google Colab with T4 GPU

First commit all project files to GitHub.

1. Create a Colab notebook.
2. Select Runtime > Change runtime type.
3. Select T4 GPU if available.
4. Run the following setup cell.

### Cell 1: clone and install

```python
import pathlib
import subprocess
import sys

REPOSITORY = "https://github.com/digimarketingai/kokoro-multilingual-tts.git"
PROJECT = pathlib.Path("/content/kokoro-multilingual-tts")

if not PROJECT.exists():
    subprocess.run(
        ["git", "clone", REPOSITORY, str(PROJECT)],
        check=True,
    )
else:
    print("Using existing project directory.")
    print("Existing files will not be deleted or overwritten.")

installer = PROJECT / "colab_setup.py"
if not installer.exists():
    raise RuntimeError(
        "colab_setup.py is missing. Upload/commit the replacement "
        "project files before running this notebook."
    )

subprocess.run(
    [sys.executable, str(installer)],
    check=True,
)
```

### Cell 2: launch

```python
import os
import pathlib
import subprocess
from getpass import getpass

PROJECT = pathlib.Path("/content/kokoro-multilingual-tts")
PYTHON = PROJECT / ".venv-colab/bin/python"

environment = os.environ.copy()
environment["GRADIO_USERNAME"] = "kokoro"
environment["GRADIO_PASSWORD"] = getpass(
    "Choose a password for the shared app: "
)

if not environment["GRADIO_PASSWORD"]:
    raise ValueError("A non-empty password is required.")

subprocess.run(
    [
        str(PYTHON),
        str(PROJECT / "app.py"),
        "--device",
        "cuda",
        "--share",
    ],
    cwd=PROJECT,
    env=environment,
    check=True,
)
```

Open the Gradio share URL printed by the launch cell.

Login username: `kokoro`

Login password: the password entered in the cell.

The launch cell remains running while the application is active.
Stop the cell to stop the application.

The notebook kernel does not need to use Python 3.12.
The app runs using its own Python 3.12 executable.

GPU allocation and runtime duration are controlled by Colab.
Free runtimes may terminate workloads used mainly through a web UI.
This is not a permanent hosting deployment.

## Local installation

This project targets Python 3.12.

### Ubuntu / Debian system dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  espeak-ng libespeak-ng1 libsndfile1 \
  build-essential cmake pkg-config
```

### Create a virtual environment

```bash
git clone https://github.com/digimarketingai/kokoro-multilingual-tts.git
cd kokoro-multilingual-tts

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### NVIDIA GPU installation

Requires an NVIDIA GPU and a compatible driver.

```bash
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt
python -m pip check
python app.py --device cuda
```

### CPU installation

```bash
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -m pip check
python app.py --device cpu
```

Open:

```text
http://127.0.0.1:7860
```

For Japanese, the app downloads UniDic when first needed.
You can also install it in advance:

```bash
python -m unidic download
```

## Usage

1. Enter source text.
2. Choose a language or leave it on Auto.
3. Choose a voice or leave it on Auto.
4. Optionally enable Translate.
5. Select the target language and translation voice.
6. Click Generate.
7. Edit the translation if necessary.
8. Click Speak edited translation.

For multilingual dialogue:

- Enable Dialogue.
- Put one turn per line.
- Leave source language on Auto for per-line detection.
- Leave voice on Auto for alternating female/male voices.

If the selected language has no male voice, the available voice is reused.

Without Dialogue, the input is treated as one language.
For mixed-language paragraphs, separate languages into dialogue lines.

## Command-line arguments

| Argument | Default | Purpose |
|---|---|---|
| --device | auto | auto, cuda, or cpu |
| --share | off | Create a Gradio public share link |
| --host | 127.0.0.1 | Bind address |
| --port | 7860 | Server port |

Examples:

```bash
python app.py --device cuda
python app.py --device cpu --port 7861
python app.py --device cuda --share
```

## Authentication

Set both environment variables before launching:

```bash
export GRADIO_USERNAME="kokoro"
read -s -p "Password: " GRADIO_PASSWORD
export GRADIO_PASSWORD
echo

python app.py --share
```

Never commit passwords or tokens to GitHub.

## Privacy

- Speech is generated in the machine running the app.
- On Colab, that machine is a cloud runtime, not your own computer.
- Initial model, voice, and language-resource downloads need internet.
- Enabling translation sends source text to an external translation service.
- Gradio sharing makes the application reachable through a share URL.
- Authentication controls who can submit jobs.
- Audio may be stored temporarily in Gradio's file cache.
- Do not use this demo for confidential information.

## Troubleshooting

### CUDA unavailable

Select a GPU runtime, reconnect, and rerun `colab_setup.py`.

Check CUDA using the application's Python:

```bash
/content/kokoro-multilingual-tts/.venv-colab/bin/python -c \
"import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The launch command uses `--device cuda` so a missing GPU produces an
error rather than silently falling back to CPU.

### Wrong Python version

Use `.venv-colab/bin/python` for the app on Colab.

Do not launch the app using the notebook's default Python executable.

### Translation failed

Check internet access and retry later.
The translation service may throttle requests.
Source audio and successfully translated text are retained when possible.

### Japanese dictionary error

```bash
/content/kokoro-multilingual-tts/.venv-colab/bin/python \
  -m unidic download
```

### Incorrect language detection

Use complete sentences or choose the source language manually.
Short text and Japanese written entirely in kanji can be ambiguous.

### First generation is slow

The first request loads the model and may download language resources
and the selected voice.

### Port already in use

Stop the previous application process or launch with another port:

```bash
python app.py --port 7861
```

### Existing Colab checkout is old

The setup cell deliberately does not delete or overwrite your work.

If you have no uncommitted edits, update the checkout:

```bash
git -C /content/kokoro-multilingual-tts pull --ff-only
```

Then rerun setup.

## Dependency notes

- This project targets Python 3.12.
- Kokoro, Misaki, and PyTorch are pinned.
- Some other dependencies use bounded version ranges.
- requirements.txt is not a complete transitive dependency lockfile.
- Validate your environment before wider deployment.

## Credits

- Kokoro-82M by hexgrad
- Kokoro and Misaki
- PyTorch
- Gradio
- deep-translator
- langdetect
- OpenCC
- uv

## License

Application code: MIT.

Model weights and third-party dependencies retain their own licenses.
