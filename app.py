#!/usr/bin/env python3
"""Kokoro multilingual speech and translation web application."""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "This project targets Python 3.12.\n"
        "On Colab, run colab_setup.py first and use its Python executable."
    )

import gradio as gr
import numpy as np
import torch
from deep_translator import GoogleTranslator
from kokoro import KModel, KPipeline
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from opencc import OpenCC

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("kokoro_app")

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 12000
AUTO = "auto"

DetectorFactory.seed = 0
TO_SIMPLIFIED = OpenCC("t2s")
TO_TRADITIONAL = OpenCC("s2t")


@dataclass(frozen=True)
class Language:
    label: str
    code: str
    translation_code: str
    voices: tuple[str, ...]


CHINESE_VOICES = (
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)

LANGUAGES = {
    "en-us": Language(
        "English (US) · 美式英文",
        "a",
        "en",
        (
            "af_heart", "af_alloy", "af_aoede", "af_bella",
            "af_jessica", "af_kore", "af_nicole", "af_nova",
            "af_river", "af_sarah", "af_sky",
            "am_michael", "am_adam", "am_echo", "am_eric",
            "am_fenrir", "am_liam", "am_onyx", "am_puck",
            "am_santa",
        ),
    ),
    "en-gb": Language(
        "English (UK) · 英式英文",
        "b",
        "en",
        (
            "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
            "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
        ),
    ),
    "zh-tw": Language(
        "繁體中文 · Traditional Chinese",
        "z",
        "zh-TW",
        CHINESE_VOICES,
    ),
    "zh-cn": Language(
        "简体中文 · Simplified Chinese",
        "z",
        "zh-CN",
        (
            "zf_xiaoxiao", "zm_yunjian",
            *(
                voice for voice in CHINESE_VOICES
                if voice not in ("zf_xiaoxiao", "zm_yunjian")
            ),
        ),
    ),
    "ja": Language(
        "日本語 · Japanese",
        "j",
        "ja",
        ("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
    ),
    "es": Language(
        "Español · 西班牙文",
        "e",
        "es",
        ("ef_dora", "em_alex", "em_santa"),
    ),
    "fr": Language(
        "Français · 法文",
        "f",
        "fr",
        ("ff_siwis",),
    ),
    "it": Language(
        "Italiano · 義大利文",
        "i",
        "it",
        ("if_sara", "im_nicola"),
    ),
    "pt": Language(
        "Português (Brasil) · 巴西葡萄牙文",
        "p",
        "pt",
        ("pf_dora", "pm_alex", "pm_santa"),
    ),
    "hi": Language(
        "हिन्दी · 印地文",
        "h",
        "hi",
        ("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
    ),
}

LANGUAGE_CHOICES = [
    (language.label, key)
    for key, language in LANGUAGES.items()
]

DETECTED_LANGUAGE_MAP = {
    "en": "en-us",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-tw",
    "ja": "ja",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "hi": "hi",
}


def validate_text(text: str) -> str:
    text = (text or "").strip()

    if not text:
        raise gr.Error("Enter some text first. · 請先輸入文字。")

    if len(text) > MAX_TEXT_LENGTH:
        raise gr.Error(
            f"Maximum input length: {MAX_TEXT_LENGTH:,} characters."
        )

    return text


def split_text(text: str, limit: int):
    """Split at nearby punctuation or whitespace, with a hard size limit."""
    text = text.strip()

    while text:
        if len(text) <= limit:
            yield text
            break

        window = text[:limit]
        boundaries = [
            match.end()
            for match in re.finditer(r"[\s.!?。！？；;，,]", window)
        ]

        cut = boundaries[-1] if boundaries else limit
        if cut < limit // 3:
            cut = limit

        piece = text[:cut].strip()
        if piece:
            yield piece

        text = text[cut:].lstrip()


def resolve_language(text: str, selected: str):
    """Return an available speech language and an explanatory note."""
    if selected != AUTO:
        return selected, LANGUAGES[selected].label

    # Kana is a strong hint for Japanese. Otherwise prefer statistical
    # detection; Unicode blocks alone do not uniquely identify a language.
    if re.search(r"[\u3040-\u30ff\uff66-\uff9f]", text):
        detected = "ja"
    else:
        try:
            detected = detect(text)
        except LangDetectException:
            detected = "unknown"

    if detected in ("zh-cn", "zh-tw"):
        if TO_SIMPLIFIED.convert(text) != text:
            detected = "zh-tw"
        elif TO_TRADITIONAL.convert(text) != text:
            detected = "zh-cn"

    language = DETECTED_LANGUAGE_MAP.get(detected)

    if language is None:
        return (
            "en-us",
            f"Detected {detected}: unsupported for native speech; "
            "using an English voice.",
        )

    return language, f"{LANGUAGES[language].label} [auto]"


def voice_choices(language: str):
    if language == AUTO:
        voices = sorted({
            voice
            for item in LANGUAGES.values()
            for voice in item.voices
        })
    else:
        voices = list(LANGUAGES[language].voices)

    return [("Auto · 自動", AUTO)] + [(voice, voice) for voice in voices]


def refresh_voices(language: str):
    return gr.Dropdown(
        choices=voice_choices(language),
        value=AUTO,
    )


def choose_voice(language: str, requested: str, index: int):
    available = LANGUAGES[language].voices

    if requested != AUTO and requested in available:
        return requested, ""

    female = next(
        (voice for voice in available if voice[1] == "f"),
        available[0],
    )
    male = next(
        (voice for voice in available if voice[1] == "m"),
        female,
    )

    note = ""
    if requested != AUTO:
        note = f"Voice {requested} does not match this language; used Auto."

    return (female if index % 2 == 0 else male), note


def ensure_japanese_dictionary():
    import unidic

    dictionary = Path(unidic.DICDIR) / "mecabrc"

    if not dictionary.exists():
        LOGGER.info("Downloading the Japanese UniDic dictionary...")
        subprocess.run(
            [sys.executable, "-m", "unidic", "download"],
            check=True,
        )

    if not dictionary.exists():
        raise RuntimeError(
            "Japanese dictionary is missing. "
            "Run: python -m unidic download"
        )


class SpeechEngine:
    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.pipelines = {}
        self.lock = threading.RLock()

    def pipeline(self, language: str):
        code = LANGUAGES[language].code

        if code in self.pipelines:
            return self.pipelines[code]

        if code == "j":
            ensure_japanese_dictionary()

        if self.model is None:
            LOGGER.info("Loading %s on %s", MODEL_ID, self.device)
            self.model = KModel(repo_id=MODEL_ID).to(self.device).eval()

        self.pipelines[code] = KPipeline(
            lang_code=code,
            repo_id=MODEL_ID,
            model=self.model,
        )
        return self.pipelines[code]

    def non_english_audio(self, pipeline, text, voice, speed):
        """Split again if phonemization exceeds the model input limit."""
        phonemes, _ = pipeline.g2p(text)

        if not phonemes:
            return

        if len(phonemes) > 500:
            if len(text) <= 1:
                raise RuntimeError(
                    "A single character produced too many phonemes."
                )

            middle = len(text) // 2
            left_space = text.rfind(" ", 0, middle + 1)
            if left_space > len(text) // 4:
                middle = left_space

            yield from self.non_english_audio(
                pipeline, text[:middle].strip(), voice, speed
            )
            yield from self.non_english_audio(
                pipeline, text[middle:].strip(), voice, speed
            )
            return

        for result in pipeline.generate_from_tokens(
            phonemes,
            voice=voice,
            speed=speed,
        ):
            if result.audio is not None:
                yield result.audio

    def speak(
        self,
        text: str,
        selected_language: str,
        selected_voice: str,
        speed: float,
        dialogue: bool,
        pause: float,
    ):
        text = validate_text(text)

        lines = (
            [line.strip() for line in text.splitlines() if line.strip()]
            if dialogue
            else [text]
        )

        segments = []
        notes = []

        with self.lock, torch.inference_mode():
            for index, line in enumerate(lines):
                language, language_note = resolve_language(
                    line, selected_language
                )
                voice, voice_note = choose_voice(
                    language,
                    selected_voice,
                    index if dialogue else 0,
                )

                pipeline = self.pipeline(language)
                spoken_text = (
                    TO_SIMPLIFIED.convert(line)
                    if LANGUAGES[language].code == "z"
                    else line
                )

                line_segments = []

                for chunk in split_text(spoken_text, 180):
                    if LANGUAGES[language].code in ("a", "b"):
                        audio_iterator = (
                            result.audio
                            for result in pipeline(
                                chunk,
                                voice=voice,
                                speed=float(speed),
                                split_pattern=None,
                            )
                            if result.audio is not None
                        )
                    else:
                        audio_iterator = self.non_english_audio(
                            pipeline, chunk, voice, float(speed)
                        )

                    for audio in audio_iterator:
                        if isinstance(audio, torch.Tensor):
                            audio = audio.detach().float().cpu().numpy()

                        audio = np.asarray(
                            audio, dtype=np.float32
                        ).reshape(-1)

                        if audio.size:
                            line_segments.append(audio)

                if line_segments:
                    if segments and pause > 0:
                        segments.append(
                            np.zeros(
                                int(SAMPLE_RATE * pause),
                                dtype=np.float32,
                            )
                        )

                    segments.extend(line_segments)
                else:
                    notes.append(f"Line {index + 1}: no speech generated.")

                notes.append(
                    f"Line {index + 1}: {language_note}; voice={voice}"
                )
                if voice_note:
                    notes.append(voice_note)

        if not segments:
            raise RuntimeError(
                "No audio was generated. Try a full sentence and "
                "select its language manually."
            )

        waveform = np.concatenate(segments).astype(np.float32)
        waveform = np.nan_to_num(waveform)
        waveform = np.clip(waveform, -1.0, 1.0)

        return (SAMPLE_RATE, waveform), notes


def translate_text(text: str, source: str, target: str) -> str:
    source_code = (
        "auto"
        if source == AUTO
        else LANGUAGES[source].translation_code
    )
    target_code = LANGUAGES[target].translation_code

    translator = GoogleTranslator(
        source=source_code,
        target=target_code,
    )

    translated_lines = []

    # Preserve line boundaries for dialogue.
    for line in text.splitlines():
        if not line.strip():
            translated_lines.append("")
            continue

        translated_chunks = []

        for chunk in split_text(line, 4000):
            for attempt in range(3):
                try:
                    translated = translator.translate(chunk)
                    if not translated:
                        raise RuntimeError(
                            "Translation service returned empty text."
                        )
                    translated_chunks.append(translated)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))

        translated_lines.append(" ".join(translated_chunks))

    result = "\n".join(translated_lines)

    if target == "zh-tw":
        result = TO_TRADITIONAL.convert(result)
    elif target == "zh-cn":
        result = TO_SIMPLIFIED.convert(result)

    return result


def build_app(engine: SpeechEngine):
    hardware = (
        torch.cuda.get_device_name(0)
        if engine.device == "cuda"
        else "CPU"
    )

    def generate(
        source_text,
        source_language,
        source_voice,
        should_translate,
        target_language,
        target_voice,
        speed,
        dialogue,
        pause,
    ):
        source_text = validate_text(source_text)
        started = time.perf_counter()

        source_audio = None
        target_audio = None
        translated = ""
        messages = [f"Device: {hardware}"]

        try:
            source_audio, notes = engine.speak(
                source_text,
                source_language,
                source_voice,
                speed,
                dialogue,
                pause,
            )
            messages.extend(["SOURCE SPEECH", *notes])
        except Exception as exc:
            LOGGER.exception("Source speech failed")
            messages.append(f"Source speech failed: {exc}")

        if should_translate:
            try:
                translated = translate_text(
                    source_text,
                    source_language,
                    target_language,
                )
            except Exception as exc:
                LOGGER.exception("Translation failed")
                messages.append(f"Translation failed: {exc}")

            if translated:
                try:
                    target_audio, notes = engine.speak(
                        translated,
                        target_language,
                        target_voice,
                        speed,
                        dialogue,
                        pause,
                    )
                    messages.extend(["TRANSLATION SPEECH", *notes])
                except Exception as exc:
                    LOGGER.exception("Translation speech failed")
                    messages.append(f"Translation speech failed: {exc}")

        elapsed = time.perf_counter() - started
        messages.append(f"Elapsed: {elapsed:.1f} seconds")

        return (
            source_audio,
            translated,
            target_audio,
            "\n".join(messages),
        )

    def speak_edited(text, language, voice, speed, dialogue, pause):
        text = validate_text(text)

        try:
            audio, notes = engine.speak(
                text, language, voice, speed, dialogue, pause
            )
            return audio, "\n".join([f"Device: {hardware}", *notes])
        except Exception as exc:
            LOGGER.exception("Edited translation speech failed")
            return None, f"Speech failed: {exc}"

    with gr.Blocks(
        title="Kokoro Multilingual TTS",
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown(
            "# 🎙️ Kokoro Multilingual TTS + Translation\n"
            "## 多語言語音合成＋翻譯\n"
            f"**Device / 執行裝置:** {hardware}\n\n"
            "Enter text, generate speech, optionally translate, "
            "then edit and replay the translation."
        )

        gr.Markdown(
            "**Privacy / 隱私:** When Translate is enabled, your source "
            "text is sent to an external translation service. "
            "Do not submit confidential information.\n\n"
            "勾選翻譯時，原文會傳送至外部翻譯服務，請勿輸入機密資料。"
        )

        source_text = gr.Textbox(
            label="Source text (ST) · 原文",
            lines=7,
            placeholder=(
                "Hello! Welcome to our multilingual demo.\n"
                "你好，歡迎使用多語言語音合成。\n"
                "こんにちは。"
            ),
        )

        with gr.Row():
            source_language = gr.Dropdown(
                choices=[("Auto · 自動", AUTO)] + LANGUAGE_CHOICES,
                value=AUTO,
                label="Source language · 原文語言",
            )
            source_voice = gr.Dropdown(
                choices=voice_choices(AUTO),
                value=AUTO,
                label="Source voice · 原文聲音",
            )

        with gr.Row():
            should_translate = gr.Checkbox(
                value=False,
                label="🌍 Translate · 翻譯",
            )
            dialogue = gr.Checkbox(
                value=False,
                label="Dialogue: one turn per line · 對話模式",
            )

        with gr.Row():
            target_language = gr.Dropdown(
                choices=LANGUAGE_CHOICES,
                value="zh-tw",
                label="Target language · 目標語言",
            )
            target_voice = gr.Dropdown(
                choices=voice_choices("zh-tw"),
                value=AUTO,
                label="Translation voice · 譯文聲音",
            )

        with gr.Row():
            speed = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Speech speed · 語速",
            )
            pause = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=0.35,
                label="Dialogue line pause (seconds) · 對話停頓",
            )

        generate_button = gr.Button(
            "🔊 Generate · 生成",
            variant="primary",
        )

        source_audio = gr.Audio(
            label="Source speech (ST) · 原文語音",
            format="wav",
            interactive=False,
        )

        translated_text = gr.Textbox(
            label="Editable translation (TT) · 可編輯譯文",
            lines=7,
            interactive=True,
        )

        edited_button = gr.Button(
            "🔁 Speak edited translation · 朗讀修改後的譯文"
        )

        target_audio = gr.Audio(
            label="Translation speech (TT) · 譯文語音",
            format="wav",
            interactive=False,
        )

        status = gr.Textbox(
            label="Status · 執行狀態",
            lines=8,
            interactive=False,
        )

        gr.Markdown(
            "Auto language detection can be wrong, especially for "
            "short text. Select the language manually when necessary.\n\n"
            "Dialogue Auto voices alternate female/male where both "
            "are available. French has only one voice in this app."
        )

        source_language.change(
            fn=refresh_voices,
            inputs=source_language,
            outputs=source_voice,
            queue=False,
        )

        target_language.change(
            fn=refresh_voices,
            inputs=target_language,
            outputs=target_voice,
            queue=False,
        )

        generate_button.click(
            fn=generate,
            inputs=[
                source_text,
                source_language,
                source_voice,
                should_translate,
                target_language,
                target_voice,
                speed,
                dialogue,
                pause,
            ],
            outputs=[
                source_audio,
                translated_text,
                target_audio,
                status,
            ],
            concurrency_id="speech",
            concurrency_limit=1,
        )

        edited_button.click(
            fn=speak_edited,
            inputs=[
                translated_text,
                target_language,
                target_voice,
                speed,
                dialogue,
                pause,
            ],
            outputs=[target_audio, status],
            concurrency_id="speech",
            concurrency_limit=1,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Select a GPU runtime and "
            "run colab_setup.py, or use --device cpu."
        )

    if device == "cuda":
        # Exercise CUDA rather than only checking device visibility.
        probe = torch.ones(4, device="cuda")
        assert (probe + probe).sum().item() == 8
        torch.cuda.synchronize()
        del probe

    username = os.getenv("GRADIO_USERNAME")
    password = os.getenv("GRADIO_PASSWORD")

    if bool(username) != bool(password):
        raise SystemExit(
            "Set both GRADIO_USERNAME and GRADIO_PASSWORD, or neither."
        )

    authentication = (username, password) if username else None

    if args.share and authentication is None:
        LOGGER.warning(
            "Public sharing enabled without authentication. "
            "Anyone with the link can submit jobs."
        )

    LOGGER.info(
        "Python=%s | Torch=%s | CUDA runtime=%s | Device=%s",
        sys.version.split()[0],
        torch.__version__,
        torch.version.cuda,
        device,
    )

    engine = SpeechEngine(device)
    demo = build_app(engine)
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=authentication,
        show_error=False,
    )


if __name__ == "__main__":
    main()
