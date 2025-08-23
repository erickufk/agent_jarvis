"""agent_core.py
Core agent logic: model factory, prompt, and run wrapper.
UI imports from here.
"""
import os, io, contextlib, traceback, json
from typing import Tuple
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from smolagents.models import OpenAIServerModel

from app_config import TOOLS_CAPABLE

import tools_local as T
from app_config import (
    PREFERRED_BACKEND,
    MODEL_BY_BACKEND,
    # legacy defaults kept for UI back-compat (not used below)
    DEFAULT_BACKEND, DEFAULT_API_MODEL, DEFAULT_HF_MODEL, DEFAULT_LOCAL_MODEL,
)

# Load .env from this directory (robust if app is launched elsewhere)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

SMART_PROMPT = """
You are a local code-and-tools agent. Assume the user is non-technical.
When instructions are ambiguous or risky, ASK up to 3 short clarifying questions in a [CLARIFY] block, then wait.

GENERAL
- Operate ONLY inside the selected action folder (use `cwd()` to confirm).
- Prefer safe tools: list_dir, read_text, search_text, write_file. Use sh only when necessary.
- Before risky actions, outline your plan in 1-3 bullets; prefer idempotent commands.
- Prefer preview-then-write: propose changes first; write only after the user says “OK”.

DOCUMENT/REPO ANALYSIS LOOP
1) Map files/folders and why they matter.
2) Use search_text to find relevant lines (show patterns).
3) Read minimal context with read_text / read_pdf / read_docx / read_xlsx / read_csv / OCR tools as needed.
4) Synthesize with file:line refs.
5) Propose changes; apply with write_file (include a brief header in new files).

ALWAYS END WITH:
[SUMMARY] one paragraph
[COMMANDS RUN] bullets
[FILES CHANGED] bullets (path → note)
[NEXT ACTIONS] bullets (if any)
"""

# ---- Helpers -----------------------------------------------------------------

def _env_clean(name: str, *fallbacks: str) -> str:
    for key in (name, *fallbacks):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    return ""

def _resolve_model_id(backend: str, incoming: str | None) -> str:
    """
    Return the model id to use for this backend.
    If the UI passed an empty/None value (or explicit '<auto>'), fall back to MODEL_BY_BACKEND.
    """
    if incoming and incoming.strip() and incoming.strip().lower() not in {"<auto>", "auto"}:
        return incoming.strip()
    # fall back to mapping; if backend missing, fall back to the preferred backend's model
    return MODEL_BY_BACKEND.get(backend) or MODEL_BY_BACKEND[PREFERRED_BACKEND]

# ---- HF via OpenAI-compatible client -----------------------------------------

class HFOpenAIModel(OpenAIServerModel):
    """
    Hugging Face Router via OpenAI-compatible client.
    Ensures `usage` is present so smolagents never sees None.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.client = self.client.with_options(extra_body={"usage": {"include": True}})
        except Exception:
            pass

# ---- Model factory ------------------------------------------------------------

def build_model(backend: str, model_id: str):
    """
    Build the correct model per provider. We request token usage so smolagents
    can log it and avoid usage=None crashes.
    """
    if backend == "API (OpenRouter)":
        key = _env_clean("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
        return LiteLLMModel(
            model_id=model_id,
            api_base="https://openrouter.ai/api/v1",
            api_key=key,
            temperature=0.2,
            max_tokens=4096,
            extra_body={"usage": {"include": True}},
        )

    if backend == "API (Hugging Face)":
        key = _env_clean("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if not key:
            raise RuntimeError("HF_TOKEN is missing in .env")
        # Use the OpenAI-compatible route exactly as HF docs show
        return HFOpenAIModel(
            model_id=model_id,                             # e.g. "openai/gpt-oss-20b"
            api_base="https://router.huggingface.co/v1",  # include /v1
            api_key=key,
            temperature=0.2,
            max_tokens=4096,
        )

    if backend == "Local (Ollama)":
        # Using Ollama's OpenAI-compatible endpoint; must include /v1.
        base = os.getenv("OLLAMA_OPENAI_BASE", "http://127.0.0.1:11434/v1")
        return LiteLLMModel(
            model_id=model_id,                       # must match `ollama list` (e.g., "gpt-oss:20b")
            api_base=base,
            api_key="ollama",                        # dummy, required by client
            temperature=0.2,
            max_tokens=4096,
            custom_llm_provider="openai",            # treat Ollama as OpenAI server
            extra_body={"usage": {"include": True}},
        )

    raise ValueError(f"Unknown backend: {backend}")

# ---- Agent cache / run wrappers -----------------------------------------------

_AGENT_CACHE = {}

def probe_model_once(backend: str, model_id: str | None) -> str:
    try:
        resolved = _resolve_model_id(backend, model_id)
        m = build_model(backend, resolved)
        txt = m.generate("ping: reply with 'pong' only.")
        return f"probe ok: {txt[:80]!r}"
    except Exception as e:
        return f"probe failed: {e}"


def get_agent(backend: str, model_id: str | None, safe_mode: bool) -> CodeAgent:
    resolved = _resolve_model_id(backend, model_id)
    key = (backend, resolved, bool(safe_mode))
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        model = build_model(backend, resolved)

        use_tools = TOOLS_CAPABLE.get(backend, True)
        agent = CodeAgent(
            model=model,
            tools=T.get_tools(safe_mode) if use_tools else [],
            add_base_tools=use_tools,      # don't advertise base tools if disabled
            instructions=SMART_PROMPT,
            planning_interval=3,
        )
        _AGENT_CACHE[key] = agent
    return agent




def run_agent_message(
    message: str,
    backend: str,
    model_id: str | None,
    safe_mode: bool,
    show_logs: bool,
    max_steps: int = 8,
) -> Tuple[str, str]:
    """
    Execute one chat turn and return (reply_text, logs_text).
    - Uses get_agent(...) which resolves the model per backend when model_id is None/empty.
    - Mirrors the agent's stdout tool logs into the UI when show_logs=True.
    - Limits the agent planning to `max_steps` (clamped to [1, 40]).
    """
    # Build / reuse cached agent
    agent = get_agent(backend, model_id, safe_mode)

    # Normalize + clamp step limit
    try:
        steps = int(max_steps)
    except (TypeError, ValueError):
        steps = 8
    steps = max(1, min(steps, 50))

    buf = io.StringIO()
    try:
        # Capture agent's console prints (tool calls, plans, etc.)
        with contextlib.redirect_stdout(buf):
            result = agent.run(message, reset=False, max_steps=steps)

        logs = buf.getvalue().strip()

        # Echo logs to server console (useful when running from a terminal)
        if show_logs and logs:
            print(f"\n[agent logs] (max_steps={steps})\n{logs}\n")

        # Stringify non-string results for the UI
        if result is None:
            text = "(no result)"
        elif isinstance(result, (str, bytes)):
            text = result.decode() if isinstance(result, bytes) else result
        else:
            # Be defensive with JSON encoding of arbitrary Python objects
            try:
                text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            except Exception:
                text = str(result)

        # Optionally append logs to the reply bubble
        if show_logs and logs:
            text = f"{text}\n\n---\n[agent logs]\n{logs}"

        return text, ""  # (reply_text, logs_text placeholder)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return f"Error: {e}\n\nTraceback:\n{tb}", ""


