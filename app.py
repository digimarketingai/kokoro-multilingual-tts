"""
Kokoro Multilingual TTS + Translation — Gradio Web UI
Kokoro 多語言語音合成＋翻譯 — Gradio 網頁介面

Enter source text (ST), optionally translate it into target text (TT),
and hear both read aloud with natural voices.
輸入原文（ST），可選擇翻譯成譯文（TT），並朗讀原文與譯文。
"""

import argparse
import os
import re
import sys

import gradio as gr
import numpy as np
import torch
from deep_translator import GoogleTranslator
from kokoro import KModel, KPipeline
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # consistent detection results / 讓偵測結果一致

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILENCE = np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.float32)  # pause between lines / 句間停頓
MAX_CHARS = 4500  # max characters per translation request / 每次翻譯請求的字數上限

# Kokoro language code -> (display name, voices; first voice = default)
# Kokoro 語言代碼 -> (顯示名稱, 聲音列表；第一個為預設)
LANGUAGES = {
    "a": ("English (US) 美式英文", ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_fenrir", "am_puck"]),
    "b": ("English (UK) 英式英文", ["bf_emma", "bf_isabella", "bm_george", "bm_fable"]),
    "e": ("Español 西班牙文", ["ef_dora", "em_alex"]),
    "f": ("Français 法文", ["ff_siwis"]),
    "h": ("हिन्दी 印地文", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    "i": ("Italiano 義大利文", ["if_sara", "im_nicola"]),
    "j": ("日本語 日文", ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jm_kumo"]),
    "p": ("Português (BR) 葡萄牙文", ["pf_dora", "pm_alex"]),
    "z": ("中文（普通話）Mandarin Chinese", ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunjian", "zm_yunxi"]),
}

# langdetect result -> Kokoro language code / langdetect 結果 -> Kokoro 語言代碼
DETECT_TO_CODE = {
    "en": "a", "es": "e", "fr": "f", "hi": "h", "it": "i",
    "ja": "j", "pt": "p", "zh-cn": "z", "zh-tw": "z",
}
DEFAULT_CODE = "a"

# Translation target -> (display name, Google Translate code, Kokoro language code)
# 翻譯目標 -> (顯示名稱, Google 翻譯代碼, Kokoro 語言代碼)
TARGETS = {
    "zh-TW": ("繁體中文 Traditional Chinese", "zh-TW", "z"),
    "zh-CN": ("简体中文 Simplified Chinese", "zh-CN", "z"),
    "en-US": ("English (US) 美式英文", "en", "a"),
    "en-GB": ("English (UK) 英式英文", "en", "b"),
    "ja": ("日本語 日文", "ja", "j"),
    "es": ("Español 西班牙文", "es", "e"),
    "fr": ("Français 法文", "fr", "f"),
    "it": ("Italiano 義大利文", "it", "i"),
    "pt": ("Português 葡萄牙文", "pt", "p"),
    "hi": ("हिन्दी 印地文", "hi", "h"),
}
DEFAULT_TARGET = "zh-TW"

KANA = re.compile(r"[\u3040-\u30ff]")              # Japanese hiragana/katakana / 日文假名
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")  # Chinese characters / 漢字

MODEL = None
PIPELINES = {}


# ---------------------------------------------------------------------------
# Model & pipelines / 模型與管線
# ---------------------------------------------------------------------------
def get_model():
    """Load the Kokoro model once and share it across languages.
    只載入一次模型，所有語言共用。"""
    global MODEL
    if MODEL is None:
        MODEL = KModel(repo_id=REPO_ID).to(DEVICE).eval()
    return MODEL


def get_pipeline(code):
    """Create (and cache) one pipeline per language.
    每種語言建立一個管線並快取。"""
    if code not in PIPELINES:
        PIPELINES[code] = KPipeline(lang_code=code, repo_id=REPO_ID, model=get_model())
    return PIPELINES[code]


# ---------------------------------------------------------------------------
# Language detection / 語言偵測
# ---------------------------------------------------------------------------
def detect_language(text):
    """Return a Kokoro language code. Unsupported languages fall back to English.
    回傳 Kokoro 語言代碼；不支援的語言改用英文。"""
    if KANA.search(text):
        return "j"
    if HAN.search(text):
        return "z"
    try:
        return DETECT_TO_CODE.get(detect(text), DEFAULT_CODE)
    except LangDetectException:
        return DEFAULT_CODE


# ---------------------------------------------------------------------------
# Translation / 翻譯
# ---------------------------------------------------------------------------
def split_lines(text):
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def translate_lines(lines, target_key):
    """Translate line by line (keeps dialogue structure). Needs internet.
    逐行翻譯（保留對話結構），需要網路連線。"""
    translator = GoogleTranslator(source="auto", target=TARGETS[target_key][1])
    results = []
    try:
        for line in lines:
            pieces = [line[i:i + MAX_CHARS] for i in range(0, len(line), MAX_CHARS)]
            results.append(" ".join((translator.translate(p) or "").strip() for p in pieces).strip())
    except Exception as e:
        raise gr.Error(f"翻譯失敗，請檢查網路連線 · Translation failed, please check your internet connection. ({e})")
    return results


# ---------------------------------------------------------------------------
# Speech synthesis / 語音合成
# ---------------------------------------------------------------------------
def to_numpy(audio):
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "cpu"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def synthesize(text, code, voice, speed):
    pipeline = get_pipeline(code)
    chunks = [to_numpy(audio) for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    chunks = [c for c in chunks if c.size]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def speak(segments, language, voice, speed):
    """Speak a list of segments. Returns (audio or None, table rows, seconds).
    朗讀多個段落，回傳（音訊或 None, 表格列, 秒數）。"""
    parts, rows = [], []
    for i, segment in enumerate(segments, 1):
        if voice != "auto":                 # manual voice decides the language / 手動聲音決定語言
            code, chosen_voice = voice[0], voice
        else:
            code = detect_language(segment) if language == "auto" else language
            chosen_voice = LANGUAGES[code][1][0]

        audio = synthesize(segment, code, chosen_voice, speed)
        if audio.size == 0:
            continue
        parts += [audio, SILENCE]

        preview = segment.replace("|", "\\|").replace("\n", " ")
        preview = preview[:40] + ("…" if len(preview) > 40 else "")
        rows.append(f"| {i} | {LANGUAGES[code][0]} | `{chosen_voice}` | {preview} |")

    if not parts:
        return None, rows, 0.0
    final_audio = np.concatenate(parts[:-1])  # drop trailing silence / 去掉最後的停頓
    return (SAMPLE_RATE, final_audio), rows, len(final_audio) / SAMPLE_RATE


def table(rows):
    header = ["| # | 語言 Language | 聲音 Voice | 文字 Text |", "|---|---|---|---|"]
    return "\n".join(header + rows)


def speak_translation(tt_text, target, tt_voice, speed, per_line):
    """Speak the target text (TT) in the target language.
    以目標語言朗讀譯文（TT）。"""
    lines = split_lines(tt_text)
    if not lines:
        return None, "⚠️ 沒有譯文 · No translation."
    code = TARGETS[target][2]
    audio, rows, seconds = speak(lines if per_line else ["\n".join(lines)], code, tt_voice, speed)
    log = "\n\n".join([
        f"#### 🅱️ 譯文 Target text (TT) → {TARGETS[target][0]}",
        table(rows),
        f"⏱️ 長度 Duration: **{seconds:.1f}s**",
    ])
    return audio, log


# ---------------------------------------------------------------------------
# Main actions / 主要功能
# ---------------------------------------------------------------------------
def generate(text, language, voice, speed, per_line, do_translate, target, tt_voice):
    text = (text or "").strip()
    if not text:
        raise gr.Error("請先輸入文字 · Please enter some text first.")

    lines = split_lines(text)

    # 1) Source text speech / 原文語音
    st_audio, st_rows, st_seconds = speak(lines if per_line else [text], language, voice, speed)
    if st_audio is None:
        raise gr.Error("無法產生語音，請換一段文字 · Could not generate audio, please try other text.")

    log = [
        "### ✅ 完成 · Done",
        "#### 🅰️ 原文 Source text (ST)",
        table(st_rows),
        f"⏱️ 長度 Duration: **{st_seconds:.1f}s**",
    ]

    # 2) Translation + target text speech / 翻譯＋譯文語音
    tt_text, tt_audio = "", None
    if do_translate:
        tt_text = "\n".join(translate_lines(lines, target))
        tt_audio, tt_log = speak_translation(tt_text, target, tt_voice, speed, per_line)
        log.append(tt_log)

    log.append(f"💻 裝置 Device: **{DEVICE}**")
    return st_audio, tt_text, tt_audio, "\n\n".join(log)


def respeak(tt_text, target, tt_voice, speed, per_line):
    """Speak an edited translation again. / 重新朗讀修改後的譯文。"""
    if not (tt_text or "").strip():
        raise gr.Error("沒有譯文可朗讀 · No translation to speak.")
    audio, log = speak_translation(tt_text, target, tt_voice, speed, per_line)
    if audio is None:
        raise gr.Error("無法產生語音 · Could not generate audio.")
    return audio, "### 🔁 已重新朗讀譯文 · Translation re-spoken\n\n" + log


# ---------------------------------------------------------------------------
# UI helpers / 介面輔助
# ---------------------------------------------------------------------------
LANGUAGE_CHOICES = [("🌐 自動偵測 · Auto-detect", "auto")] + [(name, code) for code, (name, _) in LANGUAGES.items()]
TARGET_CHOICES = [(name, key) for key, (name, _, _) in TARGETS.items()]


def voice_choices(language):
    codes = list(LANGUAGES) if language == "auto" else [language]
    choices = [("✨ 自動 · Auto", "auto")]
    for code in codes:
        name, voices = LANGUAGES[code]
        for v in voices:
            gender = "女 F" if v[1] == "f" else "男 M"
            choices.append((f"{v}（{gender}）· {name}", v))
    return choices


def on_language_change(language):
    return gr.update(choices=voice_choices(language), value="auto")


def on_target_change(target):
    return gr.update(choices=voice_choices(TARGETS[target][2]), value="auto")


def on_translate_toggle(enabled):
    return gr.update(visible=enabled), gr.update(visible=enabled)


EXAMPLES = [
    ["Hello! Welcome to this multilingual text-to-speech demo."],
    ["今天天氣很好，我們一起去公園散步吧。"],
    ["Hello, nice to meet you!\n你好，很高興認識你！\nBonjour, enchanté !\nこんにちは、よろしくお願いします。\nHola, ¡mucho gusto!"],
    ["Digital marketing helps small businesses reach more customers online."],
    ["Ciao! Come stai oggi?"],
]


def build_ui():
    with gr.Blocks(title="Kokoro 多語言語音合成＋翻譯 · Multilingual TTS + Translation") as demo:
        gr.Markdown(
            "# 🎙️ Kokoro 多語言語音合成＋翻譯 · Multilingual TTS + Translation\n"
            "輸入**原文（ST）**，系統會自動偵測語言並朗讀。勾選「翻譯」即可產生**譯文（TT）**並朗讀譯文。\n\n"
            "Enter **source text (ST)**. It auto-detects the language and reads it aloud. "
            "Tick **Translate** to get the **target text (TT)** and hear it too."
        )

        with gr.Row():
            # ---------------- Left: inputs / 左側：輸入 ----------------
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="🅰️ 原文 · Source text (ST)",
                    placeholder="在此輸入文字，每行一句… · Type here, one sentence per line…",
                    lines=8,
                )
                with gr.Row():
                    language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="auto", label="原文語言 · Source language")
                    voice = gr.Dropdown(choices=voice_choices("auto"), value="auto", label="原文聲音 · Source voice")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="語速 · Speed")
                per_line = gr.Checkbox(
                    value=True,
                    label="逐行處理（對話模式）· Process line by line (dialogue mode)",
                )

                do_translate = gr.Checkbox(value=False, label="🌍 翻譯並朗讀譯文 · Translate & speak the translation")
                with gr.Group(visible=False) as tt_settings:
                    with gr.Row():
                        target = gr.Dropdown(choices=TARGET_CHOICES, value=DEFAULT_TARGET, label="目標語言 · Target language")
                        tt_voice = gr.Dropdown(
                            choices=voice_choices(TARGETS[DEFAULT_TARGET][2]),
                            value="auto",
                            label="譯文聲音 · Translation voice",
                        )

                with gr.Row():
                    clear_btn = gr.Button("🧹 清除 · Clear")
                    run_btn = gr.Button("🔊 生成 · Generate", variant="primary")

            # ---------------- Right: outputs / 右側：輸出 ----------------
            with gr.Column(scale=2):
                st_audio_out = gr.Audio(label="🅰️ 原文語音 · Source speech (ST)", type="numpy", autoplay=True)
                with gr.Group(visible=False) as tt_outputs:
                    tt_text_out = gr.Textbox(
                        label="🅱️ 譯文（可編輯）· Translation (TT, editable)",
                        lines=6,
                        interactive=True,
                    )
                    tt_audio_out = gr.Audio(label="🅱️ 譯文語音 · Translation speech (TT)", type="numpy")
                    respeak_btn = gr.Button("🔁 朗讀修改後的譯文 · Speak edited translation")
                log_out = gr.Markdown()

        gr.Examples(examples=EXAMPLES, inputs=[text], label="範例 · Examples")

        gr.Markdown(
            "---\n"
            "支援語音 Speech: English · 中文（普通話）· 日本語 · Español · Français · Italiano · Português · हिन्दी  \n"
            "翻譯需要網路連線 · Translation requires an internet connection.  \n"
            "模型 Model: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) · "
            "翻譯 Translation: [deep-translator](https://pypi.org/project/deep-translator/) (Google Translate)"
        )

        # ---------------- Events / 事件 ----------------
        language.change(on_language_change, inputs=language, outputs=voice)
        target.change(on_target_change, inputs=target, outputs=tt_voice)
        do_translate.change(on_translate_toggle, inputs=do_translate, outputs=[tt_settings, tt_outputs])

        run_btn.click(
            generate,
            inputs=[text, language, voice, speed, per_line, do_translate, target, tt_voice],
            outputs=[st_audio_out, tt_text_out, tt_audio_out, log_out],
        )
        respeak_btn.click(
            respeak,
            inputs=[tt_text_out, target, tt_voice, speed, per_line],
            outputs=[tt_audio_out, log_out],
        )
        clear_btn.click(
            lambda: ("", None, "", None, ""),
            inputs=None,
            outputs=[text, st_audio_out, tt_text_out, tt_audio_out, log_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point / 程式進入點
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kokoro Multilingual TTS + Translation · 多語言語音合成＋翻譯")
    parser.add_argument("--share", action="store_true", help="建立公開分享連結 · Create a public share link")
    parser.add_argument("--host", default="127.0.0.1", help="伺服器位址 · Server host")
    parser.add_argument("--port", type=int, default=7860, help="連接埠 · Server port")
    args = parser.parse_args()

    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    share = args.share or in_colab  # Colab always needs a share link / Colab 需要分享連結

    print(f"⏳ 載入模型中… Loading model on {DEVICE}…")
    get_pipeline(DEFAULT_CODE)  # warm up / 預先載入
    print("✅ 模型已就緒 · Model ready!")

    build_ui().queue().launch(share=share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
