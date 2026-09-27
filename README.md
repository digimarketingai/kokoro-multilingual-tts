# 🎙️ Kokoro Multilingual TTS + Translation · Kokoro 多語言語音合成＋翻譯

A simple Gradio web app. Enter **source text (ST)** and it **auto-detects the language** and reads it aloud. Tick **Translate** to also get the **target text (TT)** and hear it spoken. Speech is powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). The interface is bilingual (Traditional Chinese / English).

一個簡單的 Gradio 網頁工具：輸入**原文（ST）**，系統會**自動偵測語言**並朗讀；勾選「翻譯」即可產生**譯文（TT）**並朗讀譯文。語音使用 [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) 模型，介面為繁體中文／英文雙語。

---

## ✨ Features · 功能

- 🌐 **Auto language detection** · 自動偵測語言
- 🌍 **Translation (ST → TT)** with speech for both · **翻譯（原文 → 譯文）**，原文與譯文皆可朗讀
- ✏️ **Edit the translation** and hear it again · **可編輯譯文**並重新朗讀
- 💬 **Dialogue mode**: each line gets its own language and voice · **對話模式**：每行各自偵測語言與聲音
- 🎚️ Choose language, voice, and speed manually · 可手動選擇語言、聲音、語速
- 🆓 Free, open-weight, runs locally or on Google Colab · 免費開源，可在本機或 Google Colab 執行
- 🔗 Optional public share link · 可選擇建立公開分享連結

## 🗣️ Supported languages · 支援語言

| Language 語言 | Speech 語音 | Translation target 翻譯目標 | Example voices 聲音範例 |
|---|---|---|---|
| English (US) 美式英文 | ✅ | ✅ | `af_heart`, `am_michael` |
| English (UK) 英式英文 | ✅ | ✅ | `bf_emma`, `bm_george` |
| 繁體中文 Traditional Chinese | ✅ (Mandarin 普通話) | ✅ | `zf_xiaobei`, `zm_yunxi` |
| 简体中文 Simplified Chinese | ✅ (Mandarin 普通話) | ✅ | `zf_xiaoxiao`, `zm_yunjian` |
| 日本語 Japanese | ✅ | ✅ | `jf_alpha`, `jm_kumo` |
| Español 西班牙文 | ✅ | ✅ | `ef_dora`, `em_alex` |
| Français 法文 | ✅ | ✅ | `ff_siwis` |
| Italiano 義大利文 | ✅ | ✅ | `if_sara`, `im_nicola` |
| Português 葡萄牙文 | ✅ | ✅ | `pf_dora`, `pm_alex` |
| हिन्दी 印地文 | ✅ | ✅ | `hf_alpha`, `hm_omega` |

The source text can be in almost any language when translating. Only speech is limited to the list above. Unsupported source languages are read with an English voice.
翻譯時原文幾乎可以是任何語言，只有語音受上表限制；不支援的原文語言會以英文聲音朗讀。

---

## 🚀 Quick Start · 快速開始

**Requirements · 需求:** Python 3.10 – 3.12, Git, internet (for translation · 翻譯需要網路)

### 1. Clone · 下載專案

```bash
git clone https://github.com/digimarketingai/kokoro-multilingual-tts.git
cd kokoro-multilingual-tts
```

### 2. Create & activate a virtual environment · 建立並啟用虛擬環境

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install · 安裝套件

```bash
pip install -r requirements.txt
```

Install **espeak-ng** too. Kokoro uses it for words it doesn't know · 另請安裝 **espeak-ng**（Kokoro 遇到不認識的字時會用到）:

| OS 系統 | Command 指令 |
|---|---|
| Ubuntu / Debian | `sudo apt-get install espeak-ng` |
| macOS | `brew install espeak-ng` |
| Windows | Download the `.msi` from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) · 下載 `.msi` 安裝檔 |

### 4. Run · 執行

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser. · 在瀏覽器開啟 **http://127.0.0.1:7860**。

Create a public link · 建立公開連結:
```bash
python app.py --share
```

---

## ☁️ Google Colab

Paste into one Colab cell and run. A public link will appear. · 貼到 Colab 儲存格執行，會出現公開連結。

```python
!apt-get -qq install -y espeak-ng > /dev/null
!git clone https://github.com/digimarketingai/kokoro-multilingual-tts.git
%cd kokoro-multilingual-tts
!pip install -q -r requirements.txt
!python app.py --share
```

💡 Select **Runtime → Change runtime type → T4 GPU** for faster speech. · 選擇 **執行階段 → 變更執行階段類型 → T4 GPU** 速度更快。

---

## 🧭 How to use · 使用方式

1. Type or paste the **source text (ST)**. Put one sentence per line for a dialogue. · 輸入**原文（ST）**，對話請每行一句。
2. Leave **Source language** and **Source voice** on *Auto*, or pick them manually. · 原文語言與聲音可保持「自動」，或手動選擇。
3. *(Optional · 選用)* Tick **🌍 Translate**, then choose a **Target language** and **Translation voice**. · 勾選 **🌍 翻譯**，選擇**目標語言**與**譯文聲音**。
4. Click **🔊 Generate**. You get · 點擊 **🔊 生成**，會得到:
   - 🅰️ Source speech (ST) · 原文語音
   - 🅱️ Translation text (TT) · 譯文
   - 🅱️ Translation speech (TT) · 譯文語音
5. *(Optional · 選用)* Edit the translation, then click **🔁 Speak edited translation**. · 修改譯文後點擊 **🔁 朗讀修改後的譯文**。

## ⚙️ Command options · 指令參數

| Option 參數 | Description 說明 | Default 預設 |
|---|---|---|
| `--share` | Public share link · 公開分享連結 | off |
| `--host` | Server host · 伺服器位址 | `127.0.0.1` |
| `--port` | Server port · 連接埠 | `7860` |

---

## 🛠️ Troubleshooting · 疑難排解

- **The first run is slow** → The model (~330 MB) downloads once. · **第一次很慢** → 模型只需下載一次。
- **Translation failed** → Check your internet connection. The free translation service may limit very frequent requests, so wait a moment and try again. · **翻譯失敗** → 請檢查網路；免費翻譯服務可能限制頻繁請求，請稍候再試。
- **Japanese error mentioning `unidic`** → Run `python -m unidic download`. · **日文出現 `unidic` 錯誤** → 執行 `python -m unidic download`。
- **Some words are skipped** → Make sure espeak-ng is installed. · **部分字詞被略過** → 請確認已安裝 espeak-ng。
- **The wrong language is detected** → Very short text is hard to detect. Use full sentences or pick the language manually. · **語言偵測錯誤** → 太短的文字難以判斷，請輸入完整句子或手動選擇語言。

---

## 📂 Project structure · 專案結構

```
kokoro-multilingual-tts/
├── app.py            # Gradio app · 主程式
├── requirements.txt  # Dependencies · 相依套件
├── README.md
└── .gitignore
```

## 🙏 Credits · 致謝

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad (Apache-2.0)
- [deep-translator](https://pypi.org/project/deep-translator/) (Google Translate)
- [Gradio](https://gradio.app) · [langdetect](https://pypi.org/project/langdetect/)

## 📄 License · 授權

MIT License
