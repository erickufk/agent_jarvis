# agent_core.py
"""
Core agent logic: model factory, prompt, cache/reset, and run wrapper.
The Gradio UI imports helpers from here.

Backends supported:
- API (OpenRouter)  -> uses LiteLLMModel with api_base=https://openrouter.ai/api/v1
- API (Hugging Face)-> OpenAI-compatible at https://router.huggingface.co/v1
- Local (Ollama)    -> OpenAI-compatible at http://127.0.0.1:11434/v1

Environment:
- OPENROUTER_API_KEY=...
- HF_TOKEN=...  (or HUGGING_FACE_HUB_TOKEN)
- OLLAMA_OPENAI_BASE=http://127.0.0.1:11434/v1  (optional override)

Exports used by the UI:
- run_agent_message(...)
- probe_model_once(...)
- reset_agent_cache(...)
- reset_agent_memory_only(...)
"""

import os, io, contextlib, traceback, json
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from smolagents.models import OpenAIServerModel

import tools_local as T
from app_config import (
    TOOLS_CAPABLE,
    PREFERRED_BACKEND,
    MODEL_BY_BACKEND,
    # kept for possible legacy UI references (not used directly here)
    DEFAULT_BACKEND, DEFAULT_API_MODEL, DEFAULT_HF_MODEL, DEFAULT_LOCAL_MODEL,
)

# Load .env from the project directory (robust if app is launched elsewhere)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# --------------------------------------------------------------------------- #
# System prompt (kept concise; agent adds reasoning in logs)
# --------------------------------------------------------------------------- #
SMART_PROMPT = """
You are Jarvis, a local code-and-tools agent helping a non-technical user.

SAFETY & SCOPE
- Operate ONLY inside the selected action folder (confirm with cwd()).
- Prefer safe tools: list_dir, read_text, search_text, write_file.
- Use sh only when necessary; outline the plan before risky actions.

WORKFLOW
1) Explore minimally (list_dir / counts) to locate relevant files.
2) search_text for specific patterns; then read just the needed snippets.
3) Synthesize findings with file:line references.
4) Propose changes; preview first; write_file only after user confirmation unless explicitly asked.

DOCUMENTS
- Prefer read_pdf_auto (it falls back to OCR), read_docx, read_xlsx, read_csv.
- Never call python_interpreter.

If instructions are ambiguous or risky, ASK up to 3 short clarifying questions in a [CLARIFY] block, then wait.

ALWAYS END WITH:
[SUMMARY] one paragraph
[COMMANDS RUN] bullets
[FILES CHANGED] bullets (path → note)
[NEXT ACTIONS] bullets
"""

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _env_clean(name: str, *fallbacks: str) -> str:
    """Return first non-empty env var among name and fallbacks."""
    for key in (name, *fallbacks):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    return ""

def _resolve_model_id(backend: str, incoming: Optional[str]) -> str:
    """
    Resolve the model name for a backend.
    If incoming is empty/<auto>, use MODEL_BY_BACKEND mapping.
    """
    if incoming and incoming.strip() and incoming.strip().lower() not in {"<auto>", "auto"}:
        return incoming.strip()
    mapped = MODEL_BY_BACKEND.get(backend)
    if mapped:
        return mapped
    # fallback to preferred backend mapping if key missing
    return MODEL_BY_BACKEND[PREFERRED_BACKEND]

# --------------------------------------------------------------------------- #
# Hugging Face (OpenAI-compatible) wrapper to ensure usage attribution exists
# --------------------------------------------------------------------------- #
class HFOpenAIModel(OpenAIServerModel):
    """
    Route via Hugging Face's OpenAI-compatible endpoint.
    Ensure responses include `usage` so smolagents won't see None.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # Ask server to include token usage stats if supported
            self.client = self.client.with_options(extra_body={"usage": {"include": True}})
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_model(backend: str, model_id: str):
    """
    Build a smolagents Model for the selected backend + model_id.
    We request token usage to avoid crashes when usage=None.
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
        return HFOpenAIModel(
            model_id=model_id,                             # e.g. "openai/gpt-oss-20b"
            api_base="https://router.huggingface.co/v1",  # include /v1 per HF docs
            api_key=key,
            temperature=0.2,
            max_tokens=4096,
        )

    if backend == "Local (Ollama)":
        # OpenAI-compatible route exposed by Ollama. Must include /v1
        base = os.getenv("OLLAMA_OPENAI_BASE", "http://127.0.0.1:11434/v1")
        return LiteLLMModel(
            model_id=model_id,                 # must match `ollama list` name, e.g. "gpt-oss:20b"
            api_base=base,
            api_key="ollama",                  # dummy value; required by client
            temperature=0.2,
            max_tokens=4096,
            custom_llm_provider="openai",      # treat Ollama as OpenAI server
            extra_body={"usage": {"include": True}},
        )

    raise ValueError(f"Unknown backend: {backend}")

# --------------------------------------------------------------------------- #
# Agent cache with instance metadata (id/created/steps)
# --------------------------------------------------------------------------- #
# key = (backend: str, resolved_model_id: str, safe_mode: bool)
_AGENT_CACHE: Dict[tuple, Dict[str, Any]] = {}

def _cache_get(key: tuple):
    entry = _AGENT_CACHE.get(key)
    return entry["agent"] if entry else None

def _cache_put(key: tuple, agent: CodeAgent):
    _AGENT_CACHE[key] = {
        "agent": agent,
        "id": str(uuid4()),
        "created": datetime.now(),
        "steps": 0,
    }

def current_agent_meta(backend: str, model_id: Optional[str], safe_mode: bool):
    """Return metadata for the currently cached agent entry, or None."""
    resolved = _resolve_model_id(backend, model_id)
    key = (backend, resolved, bool(safe_mode))
    entry = _AGENT_CACHE.get(key)
    if not entry:
        return None
    return {
        "id": entry["id"],
        "created": entry["created"].isoformat(timespec="seconds"),
        "steps": entry["steps"],
        "model": resolved,
        "backend": backend,
        "safe_mode": bool(safe_mode),
    }

# --------------------------------------------------------------------------- #
# Probe and factory
# --------------------------------------------------------------------------- #
def probe_model_once(backend: str, model_id: Optional[str]) -> str:
    """
    Quick connectivity test; returns a short string indicating success/failure.
    """
    try:
        resolved = _resolve_model_id(backend, model_id)
        m = build_model(backend, resolved)
        res = m.generate("ping: reply with 'pong' only.")
        # m.generate(...) may return a ChatMessage or a string; normalize
        txt = res if isinstance(res, str) else getattr(res, "content", str(res))
        return f"probe ok: {str(txt)[:80]!r}"
    except Exception as e:
        return f"probe failed: {e}"

def get_agent(backend: str, model_id: Optional[str], safe_mode: bool) -> CodeAgent:
    resolved = _resolve_model_id(backend, model_id)
    key = (backend, resolved, bool(safe_mode))
    agent = _cache_get(key)
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
        _cache_put(key, agent)
    return agent

# --------------------------------------------------------------------------- #
# Cache reset helpers
# --------------------------------------------------------------------------- #
def reset_agent_cache(
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    safe_mode: Optional[bool] = None
) -> str:
    """
    Clear cached agent(s).
    - If backend/model/safe_mode are provided: clear that entry.
    - If nothing is provided: clear all agents.
    Returns a short status string.
    """
    global _AGENT_CACHE

    # Clear all
    if backend is None and model_id is None and safe_mode is None:
        n = len(_AGENT_CACHE)
        _AGENT_CACHE.clear()
        return f"cleared all agents ({n})"

    try:
        resolved = _resolve_model_id(backend, model_id)  # type: ignore[arg-type]
        key = (backend, resolved, bool(safe_mode))
    except Exception:
        n = len(_AGENT_CACHE)
        _AGENT_CACHE.clear()
        return f"cleared all agents ({n})"

    entry = _AGENT_CACHE.pop(key, None)
    return f"cleared 1 agent (id={entry['id']})" if entry else "no cached agent to clear"

def reset_agent_memory_only(backend: str, model_id: Optional[str], safe_mode: bool) -> str:
    """
    Keep the instance in cache but call its reset() method if available.
    """
    resolved = _resolve_model_id(backend, model_id)
    key = (backend, resolved, bool(safe_mode))
    entry = _AGENT_CACHE.get(key)
    if not entry:
        return "no cached agent"
    agent = entry["agent"]
    if hasattr(agent, "reset"):
        try:
            agent.reset()
            entry["steps"] = 0
            entry["created"] = datetime.now()
            entry["id"] = str(uuid4())
            return "agent memory reset"
        except Exception as e:
            return f"reset failed: {e}"
    return "agent has no reset()"

# --------------------------------------------------------------------------- #
# Run wrapper (one turn)
# --------------------------------------------------------------------------- #
def run_agent_message(
    message: str,
    backend: str,
    model_id: Optional[str],
    safe_mode: bool,
    show_logs: bool,
    max_steps: int = 8,
) -> Tuple[str, str]:
    """
    Execute one chat turn and return (reply_text, logs_text).
    - Resolves model per backend when model_id is None/auto.
    - Mirrors the agent's stdout tool logs into the reply when show_logs=True.
    - Limits planning to `max_steps` (clamped to [1, 50]).
    """
    agent = get_agent(backend, model_id, safe_mode)

    # Clamp step limit
    try:
        steps = int(max_steps)
    except (TypeError, ValueError):
        steps = 8
    steps = max(1, min(steps, 50))

    # Snapshot metadata before run
    meta_before = current_agent_meta(backend, model_id, safe_mode)
    header = ""
    if meta_before:
        header = f"[agent id {meta_before['id']} | created {meta_before['created']} | steps {meta_before['steps']}]"

    buf = io.StringIO()
    try:
        # Capture agent prints (tool calls etc.)
        with contextlib.redirect_stdout(buf):
            result = agent.run(message, reset=False, max_steps=steps)

        logs = buf.getvalue().strip()

        # Bump step counter for this cached instance
        resolved = _resolve_model_id(backend, model_id)
        key = (backend, resolved, bool(safe_mode))
        if key in _AGENT_CACHE:
            _AGENT_CACHE[key]["steps"] += 1

        # Echo logs to server console for diagnosis
        if show_logs:
            print(f"\n{header}\n[agent logs] (max_steps={steps})\n{logs}\n")

        # Normalize result to string for the UI
        if result is None:
            text = "(no result)"
        elif isinstance(result, (str, bytes)):
            text = result.decode() if isinstance(result, bytes) else result
        else:
            try:
                text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            except Exception:
                text = str(result)

        # Optionally append logs + header to the reply bubble
        if show_logs:
            text = f"{header}\n{text}"
            if logs:
                text = f"{text}\n\n---\n[agent logs]\n{logs}"

        return text, ""  # reply_text, (logs_text placeholder)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return f"Error: {e}\n\nTraceback:\n{tb}", ""
