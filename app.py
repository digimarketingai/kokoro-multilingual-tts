#!/usr/bin/env python3
"""
Kokoro Multilingual TTS + Translation · Kokoro 多語言語音合成＋翻譯

A simple Gradio web app:
- Enter source text (ST). The language is auto-detected and read aloud.
- Tick "Translate" to also get the target text (TT) and hear it spoken.
- Dialogue mode: each line gets its own language and voice.

Speech: Kokoro-82M (hexgrad, Apache-2.0)
Translation: deep-translator (Google Translate)
"""

import argparse
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Python version check (Kokoro supports Python 3.10 – 3.12 only)
# ---------------------------------------------------------------------------
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    sys.exit(
        f"❌ Python {sys.version.split()[0]} detected. Kokoro needs Python 3.10–3.12.\n"
        "   偵測到的 Python 版本不支援，Kokoro 需要 Python 3.10–3.12。\n"
        "   Colab: see the README 'Google Colab' section (uses uv + Python 3.12)."
    )

import numpy as np
import torch
import gradio as gr
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from kokoro import KModel, KPipeline

# Optional: Traditional ⇄ Simplified Chinese conversion (better Mandarin G2P)
try:
    from opencc import OpenCC

    _T2S = OpenCC("t2s")
    _S2T = OpenCC("s2t")
except Exception:  # pragma: no cover
    _T2S = None
    _S2T = None

DetectorFactory.seed = 0  # make langdetect deterministic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
AUTO = "Auto · 自動"
LINE_PAUSE_SEC = 0.35
MAX_CHARS_PER_REQUEST = 4500  # Google Translate limit is 5000

_MANDARIN_VOICES = [
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

# id -> info
#   kokoro : Kokoro lang_code
#   gt     : Google Translate target code
#   voices : first "f" voice = default female, first "m" voice = default male
LANGS = {
    "en-us": {
        "label": "English (US) 美式英文",
        "kokoro": "a",
        "gt": "en",
        "voices": [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_michael", "am_adam", "am_echo", "am_eric", "am_fenrir",
            "am_liam", "am_onyx", "am_puck", "am_santa",
        ],
    },
    "en-gb": {
        "label": "English (UK) 英式英文",
        "kokoro": "b",
        "gt": "en",
        "voices": [
            "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
            "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
        ],
    },
    "zh-tw": {
        "label": "繁體中文 Traditional Chinese",
        "kokoro": "z",
        "gt": "zh-TW",
        "voices": ["zf_xiaobei", "zm_yunxi"]
        + [v for v in _MANDARIN_VOICES if v not in ("zf_xiaobei", "zm_yunxi")],
    },
    "zh-cn": {
        "label": "简体中文 Simplified Chinese",
        "kokoro": "z",
        "gt": "zh-CN",
        "voices": ["zf_xiaoxiao", "zm_yunjian"]
        + [v for v in _MANDARIN_VOICES if v not in ("zf_xiaoxiao", "zm_yunjian")],
    },
    "ja": {
        "label": "日本語 Japanese",
        "kokoro": "j",
        "gt": "ja",
        "voices": ["jf_alpha", "jm_kumo", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
    },
    "es": {
        "label": "Español 西班牙文",
        "kokoro": "e",
        "gt": "es",
        "voices": ["ef_dora", "em_alex", "em_santa"],
    },
    "fr": {
        "label": "Français 法文",
        "kokoro": "f",
        "gt": "fr",
        "voices": ["ff_siwis"],
    },
    "it": {
        "label": "Italiano 義大利文",
        "kokoro": "i",
        "gt": "it",
        "voices": ["if_sara", "im_nicola"],
    },
    "pt": {
        "label": "Português 葡萄牙文",
        "kokoro": "p",
        "gt": "pt",
        "voices": ["pf_dora", "pm_alex", "pm_santa"],
    },
    "hi": {
        "label": "हिन्दी 印地文",
        "kokoro": "h",
        "gt": "hi",
        "voices": ["hf_alpha", "hm_omega", "hf_beta", "hm_psi"],
    },
}

LANG_LABELS = [info["label"] for info in LANGS.values()]
LABEL_TO_ID = {info["label"]: lid for lid, info in LANGS.items()}
ALL_VOICES = list(dict.fromkeys(v for info in LANGS.values() for v in info["voices"]))

# langdetect code -> our language id
LANGDETECT_MAP = {
    "en": "en-us",
    "zh-tw": "zh-tw",
    "zh-cn": "zh-cn",
    "ja": "ja",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "hi": "hi",
}

_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def _chinese_variant(text: str) -> str:
    """Guess Traditional vs Simplified Chinese."""
    if _T2S is not None and _S2T is not None:
        if _T2S.convert(text) != text:
            return "zh-tw"  # contains Traditional-only characters
        if _S2T.convert(text) != text:
            return "zh-cn"  # contains Simplified-only characters
    try:
        raw = detect_langs(text)[0].lang
        if raw in ("zh-tw", "zh-cn"):
            return raw
    except LangDetectException:
        pass
    return "zh-tw"


def detect_language(text: str):
    """Return (language id or None, raw detected code)."""
    t = (text or "").strip()
    if not t:
        return None, ""
    if _KANA.search(t):
        return "ja", "ja"
    if _DEVANAGARI.search(t):
        return "hi", "hi"
    if _HAN.search(t):
        return _chinese_variant(t), "zh"
    try:
        raw = detect_langs(t)[0].lang
    except LangDetectException:
        return None, ""
    return LANGDETECT_MAP.get(raw), raw


def resolve_language(text: str, lang_choice: str):
    """Return (language id, human-readable note)."""
    if lang_choice and lang_choice != AUTO:
        lid = LABEL_TO_ID[lang_choice]
        return lid, f"{LANGS[lid]['label']} (manual · 手動)"
    lid, raw = detect_language(text)
    if lid is None:
        shown = raw or "unknown"
        return "en-us", f"`{shown}` unsupported → English voice · 不支援，改用英文聲音"
    return lid, f"{LANGS[lid]['label']} (auto · 自動)"


# ---------------------------------------------------------------------------
# Kokoro model & pipelines
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PIPELINES = {}


def get_model():
    global _MODEL
    if _MODEL is None:
        print(f"⏳ Loading Kokoro-82M on {DEVICE} (first run downloads ~330 MB)...")
        _MODEL = KModel(repo_id=REPO_ID).to(DEVICE).eval()
        print("✅ Kokoro model loaded.")
    return _MODEL


def ensure_unidic():
    """Japanese needs the UniDic dictionary. Download it once if missing."""
    try:
        import unidic  # noqa: WPS433
    except ImportError:
        return
    if not os.path.exists(os.path.join(unidic.DICDIR, "mecabrc")):
        print("⏳ Downloading UniDic dictionary for Japanese (one time only)...")
        subprocess.run([sys.executable, "-m", "unidic", "download"], check=False)


def get_pipeline(kokoro_code: str) -> KPipeline:
    if kokoro_code not in _PIPELINES:
        if kokoro_code == "j":
            ensure_unidic()
        print(f"⏳ Creating pipeline for lang_code='{kokoro_code}'...")
        _PIPELINES[kokoro_code] = KPipeline(
            lang_code=kokoro_code, repo_id=REPO_ID, model=get_model()
        )
    return _PIPELINES[kokoro_code]


def default_voice_pair(lang_id: str):
    voices = LANGS[lang_id]["voices"]
    female = next((v for v in voices if v[1] == "f"), voices[0])
    male = next((v for v in voices if v[1] == "m"), female)
    return female, male


def pick_voice(lang_id: str, voice_choice: str, index: int = 0) -> str:
    if voice_choice and voice_choice != AUTO:
        return voice_choice
    female, male = default_voice_pair(lang_id)
    return female if index % 2 == 0 else male


def synthesize(text: str, lang_id: str, voice: str, speed: float):
    """Return a float32 numpy array (24 kHz) or None."""
    text = (text or "").strip()
    if not text:
        return None
    if lang_id == "zh-tw" and _T2S is not None:
        text = _T2S.convert(text)  # Mandarin G2P works best with Simplified
    pipeline = get_pipeline(LANGS[lang_id]["kokoro"])
    chunks = []
    for result in pipeline(text, voice=voice, speed=float(speed), split_pattern=r"\n+"):
        audio = getattr(result, "audio", None)
        if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
            audio = result[2]
        if audio is None:
            continue
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not chunks:
        return None
    return np.concatenate(chunks)


def split_lines(text: str):
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def speak_text(text: str, lang_choice: str, voice_choice: str, speed: float, dialogue: bool):
    """Return ((sample_rate, audio) or None, list of markdown notes)."""
    lines = split_lines(text) if dialogue else [text.strip()]
    pause = np.zeros(int(SAMPLE_RATE * LINE_PAUSE_SEC), dtype=np.float32)
    pieces, notes = [], []

    for i, line in enumerate(lines):
        if not line:
            continue
        lang_id, lang_note = resolve_language(line, lang_choice)
        voice = pick_voice(lang_id, voice_choice, i if dialogue else 0)
        prefix = f"Line {i + 1} · 第 {i + 1} 行: " if dialogue else ""
        try:
            audio = synthesize(line, lang_id, voice, speed)
        except Exception as e:  # keep going with the other lines
            notes.append(f"- {prefix}⚠️ {lang_note} · `{voice}` — error: {e}")
            continue
        if audio is None:
            notes.append(f"- {prefix}⚠️ {lang_note} · `{voice}` — no audio · 無語音")
            continue
        if pieces:
            pieces.append(pause)
        pieces.append(audio)
        notes.append(f"- {prefix}{lang_note} · voice 聲音 `{voice}`")

    if not pieces:
        return None, notes
    return (SAMPLE_RATE, np.concatenate(pieces)), notes


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _translate_once(translator: GoogleTranslator, text: str, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            out = translator.translate(text)
            return out if out is not None else ""
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def _chunk_text(text: str, limit: int = MAX_CHARS_PER_REQUEST):
    """Group lines into chunks under the request size limit."""
    chunks, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:  # extremely long single line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def translate_text(text: str, target_id: str, dialogue: bool) -> str:
    translator = GoogleTranslator(source="auto", target=LANGS[target_id]["gt"])
    if dialogue:
        # Translate line by line so each line stays aligned
        return "\n".join(_translate_once(translator, line) for line in split_lines(text))
    return "\n".join(_translate_once(translator, chunk) for chunk in _chunk_text(text.strip()))


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def _section(title: str, notes):
    return f"**{title}**\n\n" + ("\n".join(notes) if notes else "- (none)")


def on_generate(st, src_lang, src_voice, speed, do_translate, tgt_lang, tgt_voice, dialogue):
    st = (st or "").strip()
    if not st:
        raise gr.Error("Please enter source text. · 請輸入原文。")

    sections = []
    try:
        src_audio, notes = speak_text(st, src_lang, src_voice, speed, dialogue)
    except Exception as e:
        raise gr.Error(f"Speech failed · 語音生成失敗: {e}")
    sections.append(_section("🅰️ Source speech (ST) · 原文語音", notes))

    tt_text, tt_audio = "", None
    if do_translate:
        target_id = LABEL_TO_ID[tgt_lang]
        try:
            tt_text = translate_text(st, target_id, dialogue)
        except Exception as e:
            gr.Warning("Translation failed. Check internet / wait and retry. · 翻譯失敗，請檢查網路或稍候再試。")
            sections.append(f"⚠️ **Translation failed · 翻譯失敗:** {e}")
            return src_audio, "", None, "\n\n".join(sections)

        if tt_text.strip():
            try:
                tt_audio, notes = speak_text(tt_text, tgt_lang, tgt_voice, speed, dialogue)
            except Exception as e:
                notes = [f"- ⚠️ error: {e}"]
            sections.append(_section("🅱️ Translation speech (TT) · 譯文語音", notes))

    return src_audio, tt_text, tt_audio, "\n\n".join(sections)


def on_speak_edited(tt, tgt_lang, tgt_voice, speed, dialogue):
    tt = (tt or "").strip()
    if not tt:
        raise gr.Error("The translation box is empty. · 譯文是空的。")
    try:
        tt_audio, notes = speak_text(tt, tgt_lang, tgt_voice, speed, dialogue)
    except Exception as e:
        raise gr.Error(f"Speech failed · 語音生成失敗: {e}")
    return tt_audio, _section("🔁 Edited translation speech · 修改後譯文語音", notes)


def update_src_voices(lang_choice):
    if not lang_choice or lang_choice == AUTO:
        choices = [AUTO] + ALL_VOICES
    else:
        choices = [AUTO] + LANGS[LABEL_TO_ID[lang_choice]]["voices"]
    return gr.update(choices=choices, value=AUTO)


def update_tgt_voices(lang_choice):
    choices = [AUTO] + LANGS[LABEL_TO_ID[lang_choice]]["voices"]
    return gr.update(choices=choices, value=AUTO)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# 🎙️ Kokoro Multilingual TTS + Translation · Kokoro 多語言語音合成＋翻譯

Enter **source text (ST)**: the language is auto-detected and read aloud. Tick **🌍 Translate** to get the **target text (TT)** and hear it too.
輸入**原文（ST）**，系統會自動偵測語言並朗讀；勾選 **🌍 翻譯** 即可產生**譯文（TT）**並朗讀。

💬 **Dialogue mode · 對話模式**: one sentence per line; each line gets its own language and voice (Auto voices alternate female / male).
每行一句，各行自動偵測語言與聲音（自動聲音會女聲／男聲交替）。
"""

EXAMPLES = [
    ["Hello! Welcome to the Kokoro multilingual text-to-speech demo.", False],
    ["今天天氣很好，我們一起去公園散步吧。", False],
    ["こんにちは、今日はいい天気ですね。", False],
    ["Good morning! How are you today?\n我很好，謝謝！你呢？\nI'm great, thanks for asking.", True],
    ["Bonjour tout le monde, comment allez-vous ?", False],
]


def build_ui():
    with gr.Blocks(title="Kokoro Multilingual TTS + Translation") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                st = gr.Textbox(
                    label="Source text (ST) · 原文",
                    lines=8,
                    placeholder="Type or paste text here… · 在此輸入或貼上文字…",
                )
                dialogue = gr.Checkbox(
                    label="💬 Dialogue mode (one line per sentence) · 對話模式（每行一句）",
                    value=False,
                )
            with gr.Column(scale=2):
                src_lang = gr.Dropdown(
                    [AUTO] + LANG_LABELS, value=AUTO, label="Source language · 原文語言"
                )
                src_voice = gr.Dropdown(
                    [AUTO] + ALL_VOICES, value=AUTO, label="Source voice · 原文聲音"
                )
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed · 語速")
                do_translate = gr.Checkbox(label="🌍 Translate · 翻譯", value=False)
                tgt_lang = gr.Dropdown(
                    LANG_LABELS,
                    value=LANGS["en-us"]["label"],
                    label="Target language · 目標語言",
                )
                tgt_voice = gr.Dropdown(
                    [AUTO] + LANGS["en-us"]["voices"],
                    value=AUTO,
                    label="Translation voice · 譯文聲音",
                )

        gen_btn = gr.Button("🔊 Generate · 生成", variant="primary")
        status = gr.Markdown()

        src_audio = gr.Audio(
            label="🅰️ Source speech (ST) · 原文語音", type="numpy", interactive=False
        )
        tt = gr.Textbox(
            label="🅱️ Translation text (TT) · 譯文 (editable · 可編輯)", lines=6
        )
        re_btn = gr.Button("🔁 Speak edited translation · 朗讀修改後的譯文")
        tt_audio = gr.Audio(
            label="🅱️ Translation speech (TT) · 譯文語音", type="numpy", interactive=False
        )

        gr.Examples(examples=EXAMPLES, inputs=[st, dialogue], label="Examples · 範例")

        src_lang.change(update_src_voices, inputs=src_lang, outputs=src_voice)
        tgt_lang.change(update_tgt_voices, inputs=tgt_lang, outputs=tgt_voice)

        gen_btn.click(
            on_generate,
            inputs=[st, src_lang, src_voice, speed, do_translate, tgt_lang, tgt_voice, dialogue],
            outputs=[src_audio, tt, tt_audio, status],
        )
        re_btn.click(
            on_speak_edited,
            inputs=[tt, tgt_lang, tgt_voice, speed, dialogue],
            outputs=[tt_audio, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Kokoro Multilingual TTS + Translation")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    get_model()  # load once at startup so the first click is faster

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
