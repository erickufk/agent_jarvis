
# Jarvis — Local, privacy-first AI agent

Jarvis is a local, file-centric AI agent. Explore folders, analyze & translate docs, extract data, generate files, and (optionally) run batch actions — via OpenRouter, Hugging Face Router, or local Ollama.

---

## Features

* **Folder sandbox** — pick an *Action Folder*; all reads/writes stay inside it.
* **Six actions** — Explore, Analyze, Translate, Extract data, Generate docs (e.g., README), Batch (shell).
* **Multi-provider** — switch between **OpenRouter**, **HF Router**, or **Ollama** (OpenAI-compatible `/v1`).
* **Rich readers** — txt/pdf/docx/xlsx/csv; OCR for scans (Tesseract + Poppler).
* **Safety first** — “Simple mode” (no shell) by default; **Max steps** slider caps agent reasoning depth.
* **Tool calling aware** — enable/disable LLM tool-calls per backend; clean fallback guidance when tools aren’t supported.

---

## Requirements

* **Python** 3.10–3.12
* Windows 10/11, macOS, or Linux
* Optional (for OCR):

  * **Tesseract** (Windows: set `TESSERACT_CMD` to `tesseract.exe` path)
  * **Poppler** (Windows: set `POPPLER_PATH` to Poppler `bin` folder)

---

## Install

```bash
# 1) clone your repo, then:
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install deps
pip install -U pip
pip install -r requirements.txt
```

**`requirements.txt` (suggested)**

```text
gradio>=4.36
python-dotenv>=1.0
smolagents>=0.6
openai>=1.30
litellm>=1.42
pypdf>=4.2
pdf2image>=1.17
pytesseract>=0.3.10
Pillow>=10.3
python-docx>=1.1
openpyxl>=3.1
chardet>=5.2
```

> If you don’t need OCR, you can omit `pdf2image` and `pytesseract`.

---

## Configure

Create **`.env`** in the project root (only set what you use):

```dotenv
# OpenRouter (if using)
OPENROUTER_API_KEY=or-...

# Hugging Face Router (if using)
HF_TOKEN=hf_...

# Ollama OpenAI-compatible base (defaults shown)
OLLAMA_OPENAI_BASE=http://127.0.0.1:11434/v1

# OCR (Windows only; set absolute paths)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\tools\poppler-24.02.0\Library\bin
OCR_LANGS=eng
```

Edit **`app_config.py`** to set provider-specific model IDs and tool capability:

```python
# Which backend the UI starts with
PREFERRED_BACKEND = "API (Hugging Face)"  # or OpenRouter / Local (Ollama)

# One model id per provider
MODEL_BY_BACKEND = {
    "API (OpenRouter)":   "openai/gpt-oss-120b",
    "API (Hugging Face)": "openai/gpt-oss-20b:fireworks-ai",  # provider-suffixed on HF
    "Local (Ollama)":     "gpt-oss:20b",                      # must match `ollama list`
}

# Backends that should send OpenAI tool-calls to the LLM
# (set True only if the chosen model reliably emits structured tool_calls)
TOOLS_CAPABLE = {
    "API (OpenRouter)":  True,
    "API (Hugging Face)": True,
    "Local (Ollama)":     False,
}

# theme CSS kept as-is
```

---

## Run

```bash
python app.py
# then open http://127.0.0.1:7860
```

1. In the top block, pick **Backend**, toggle **Simple mode**, set **Max steps**, then click **Apply settings**.
2. Pick your **Action Folder** (via explorer or by pasting a path), then click **Set action folder**.
3. Use the **six action buttons** (Explore / Analyze / Translate / Extract / Generate / Batch) to prefill a task; or type your own prompt.
4. Click **Send**.
5. Optional: tick **Show logs** to see agent steps in the reply.

> **Note:** If the current backend/model doesn’t support tool-calls, the UI warns you. In that case, paste document text (or an `ls -R`/`dir /s` file list) directly into chat and ask what you want (summarize, translate, extract, etc.). You can then save results to files.

---

## Provider notes

### Local (Ollama)

* Use the **OpenAI-compatible** path: `http://127.0.0.1:11434/v1`.
* In `agent_core.py` Jarvis treats Ollama as an OpenAI server (so chat/completions work).
* Make sure the **model tag** exists (`ollama pull gpt-oss:20b` then `ollama list`).

**If your local `gpt-oss:20b` supports tool-calling reliably:**

1. Set `TOOLS_CAPABLE["Local (Ollama)"] = True` in `app_config.py`.
2. Restart the app (or click **Apply settings** if you wired cache clearing).
3. Test with a tool prompt (e.g., “List all files using the `list_dir` tool”).

   * If you see errors like *“error parsing tool call: raw='list\_dir()' …”*, switch back to `False`.
4. Change the local model: edit app_config.py MODEL_BY_BACKEND["Local (Ollama)"] = "llama3:8b" (or any tag from ollama list, e.g., "gpt-oss:120b"). Then restart the app

### OpenRouter

* Put your key in `.env` (`OPENROUTER_API_KEY`).
* `MODEL_BY_BACKEND["API (OpenRouter)"] = "openai/gpt-oss-120b"` (or any supported model).
* This LLM model support tool calling → keep `TOOLS_CAPABLE["API (OpenRouter)"]=True`. 

### Hugging Face Router

* Use provider-suffixed models, e.g. `"openai/gpt-oss-20b:fireworks-ai"`.
* Put your token in `.env` (`HF_TOKEN`).
* If you see formatter errors (e.g., unexpected control tokens), try another model suffix or turn tools off (`False`).

---

## OCR (optional)

* Install **Tesseract** and (on Windows) set `TESSERACT_CMD` in `.env`.
* For scanned PDFs, install **Poppler** and set `POPPLER_PATH` (Windows).
* `OCR_LANGS` like `eng`, `eng+rus`.
* Use actions like *Analyze* or *Extract data* on scanned docs; Jarvis will call OCR readers when needed.

---

## UI guide

* **Apply settings** — applies backend/mode/steps and shows warnings for tool support.
* **Action Folder** — explorer + manual input; click **Set action folder** to lock it in.
* **Presets** — each button fills the message with a well-formed prompt.
* **Chat** — free-form tasks; **Show logs** to inspect the agent’s reasoning.
* **Max steps** — hard limit per run (1–50).
* **Simple mode** — when ON, shell is disabled.

---

## Common issues & fixes

* **Ollama 404**: you used the native path. Use `http://127.0.0.1:11434/v1`.
* **Tool-call parse errors** (`invalid character 'l'`…): your model returned a **textual** call instead of structured `tool_calls`. Set `TOOLS_CAPABLE[backend]=False` or switch to a function-calling model.
* **HF invalid header tokens**: wrong routed model. Use a provider-suffixed ID (e.g., `:fireworks-ai`) or change model.
* **401 Unauthorized**: check `.env` keys and whitespace.
* **Cyrillic/encoding**: readers try multiple encodings (`utf-8`, `cp1251`, `cp866`, `utf-16`, `latin-1`). If you still see mojibake, paste the text manually and specify the encoding.

---

## Project structure

```
.
├─ app.py            # Gradio UI (settings, folder, actions, chat)
├─ agent_core.py     # Model factory, agent cache, runner (max_steps, logs)
├─ app_config.py     # MODEL_BY_BACKEND, TOOLS_CAPABLE, theme
├─ tools_local.py    # Sandbox tools: list/read/search/write, OCR, CSV/XLSX/PDF
├─ .env              # keys & paths (example above)
└─ requirements.txt
```

---

## Tips

* Keep outputs under `reports/` so you can find results quickly.
* For big folders, ask the model to propose a plan (“what else do you need?”), then paste only relevant snippets into chat.
* Toggle **Show logs** to see each reasoning step (great for debugging prompts).
* If you switch models/backends often, add a small “Reset agent” button to clear agent memory or restart the app.

---

## License & credits

* Built with **smolagents**, **Gradio**, **OpenAI SDK**, and **LiteLLM**.
* OCR via **Tesseract** and **Poppler**.
* Model family: **gpt-oss** (20B/120B).
* Name & UX by Uriel Kaiia — Jarvis 🙌

---

### Quick start (TL;DR)

```bash
# Install
python -m venv .venv && . .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt

# Configure one provider in .env (OPENROUTER_API_KEY or HF_TOKEN or OLLAMA_OPENAI_BASE)
# Optional OCR: set TESSERACT_CMD, POPPLER_PATH

# Run
python app.py
# Pick backend, set Action Folder, press 'Apply settings'
# Click a preset or type a task; Send
```
