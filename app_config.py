"""app_config.py
Central configuration: provider defaults and theme.
Author: Uriel Kaiia (kaya.uf@gmail.com)

We keep a single source of truth for model names per provider in
MODEL_BY_BACKEND, and also export legacy constants for UI back-compat.
"""

# Which backend the UI should start with
PREFERRED_BACKEND = "API (OpenRouter)"   # or "API (Hugging Face)" or "Local (Ollama)"

# One model id per provider (update these to your liking)
MODEL_BY_BACKEND = {
    "API (OpenRouter)":  "openai/gpt-oss-120b",            # OpenRouter model id
    "API (Hugging Face)": "openai/gpt-oss-20b:fireworks-ai",           # HF Router model id
    "Local (Ollama)":     "gpt-oss:20b",                  # must match `ollama list`
}

# ---- Back-compat for existing UI code (safe to remove later) ----
DEFAULT_BACKEND    = PREFERRED_BACKEND
DEFAULT_API_MODEL  = MODEL_BY_BACKEND["API (OpenRouter)"]
DEFAULT_HF_MODEL   = MODEL_BY_BACKEND["API (Hugging Face)"]
DEFAULT_LOCAL_MODEL = MODEL_BY_BACKEND["Local (Ollama)"]

# OpenAI-ish dark theme (emerald accents)
CSS = """
:root, .dark {
  --primary-500: #10a37f;
  --primary-600: #0e906f;
  --primary-700: #0c7d60;
  --link-text-color: #10a37f;
}
.gradio-container { background: #111315; color: #e5e7eb; }
button, .btn-primary { border-radius: 14px; }
h1, h2, h3 { color: #e5e7eb; }
"""

# Which backends should use LLM tool-calls?
# Set to False for models/backends that don't reliably emit OpenAI tool_calls.
TOOLS_CAPABLE = {
    "API (OpenRouter)":  True,  # gpt-oss-120b on OR: usually no tool_calls
    "API (Hugging Face)": True,  # set False if your HF model also struggles
    "Local (Ollama)":     False, # gpt-oss:20b local: disable tool-calls
}