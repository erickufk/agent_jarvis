"""app.py
UI entry point (Gradio). Only UI logic lives here.
"""
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from app_config import TOOLS_CAPABLE
import json

import tools_local as T
from agent_core import run_agent_message, probe_model_once, reset_agent_cache, current_agent_meta

from app_config import (
    CSS,
    PREFERRED_BACKEND,
    MODEL_BY_BACKEND,
)

# Optional: define which backends are tool-capable (tweak in app_config if you like)
# e.g., {"API (OpenRouter)": False, "API (Hugging Face)": True, "Local (Ollama)": False}
from app_config import TOOLS_CAPABLE  # make sure this exists in app_config.py

load_dotenv()

with gr.Blocks(title="Local Task Agent", css=CSS, theme=gr.themes.Soft(primary_hue="emerald")) as demo: # pyright: ignore[reportPrivateImportUsage]
    gr.Markdown("## Local Task Agent — API or Local LLM, folder-sandboxed")

    # ── Settings (top) ──────────────────────────────────────────────────────────
    with gr.Group():
        with gr.Row():
            backend = gr.Radio(
                ["API (OpenRouter)", "API (Hugging Face)", "Local (Ollama)"],
                value=PREFERRED_BACKEND,
                label="LLM Backend",
            )
            safe_mode = gr.Checkbox(value=True, label="Simple Mode (safer: no shell)")
            max_steps = gr.Slider(1, 50, value=8, step=1, label="Max agent steps")

        # Read-only model display (no manual input)
        model_md = gr.Markdown(f"**Model:** `{MODEL_BY_BACKEND.get(PREFERRED_BACKEND)}`")
        tool_warn = gr.Markdown(visible=False)  # filled on backend change

        apply_btn = gr.Button("Apply settings", variant="secondary")

    # ── Folder selection ────────────────────────────────────────────────────────
    status_md = gr.Markdown("")
    with gr.Row():
        drive_root = str(Path.cwd().anchor or Path.cwd())
        with gr.Column(scale=3):
            explorer = gr.FileExplorer(
                label="Pick your ACTION FOLDER (click a folder or a file inside it)",
                root_dir=drive_root, file_count="single", glob="**/*", height=300
            )
            set_folder_btn = gr.Button("Set action folder")
        with gr.Column(scale=2):
            action_root_out = gr.Textbox(
                value=T.get_action_root(),
                label="Current action folder",
                interactive=False
            )
            folder_input = gr.Textbox(
                value=T.get_action_root(),
                label="Or type/paste full folder path here"
            )

    # ── Action presets (six main actions) ─────────────────────────────────────────
    with gr.Row():
        btn_explore  = gr.Button("🗂️ Explore folder")
        btn_analyze  = gr.Button("🧠 Analyze documents")
        btn_translate= gr.Button("🌐 Translate")
        btn_extract  = gr.Button("🔎 Extract data")
        btn_generate = gr.Button("📝 Generate new document")
        btn_batch    = gr.Button("⚙️ Batch actions (shell)")

    # ── Chat ───────────────────────────────────────────────────────────────────
    gr.Markdown("### Chat")
    chatbot = gr.Chatbot(label="Agent", height=420, type="messages")
    show_logs = gr.Checkbox(value=False, label="Show agent logs in replies")

    with gr.Row():
        # left: message box
        with gr.Column(scale=4):
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type a task and press Enter",
            )

        # right: 2x2 buttons
        with gr.Column(scale=2):
            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
            with gr.Row():
                reset_btn = gr.Button("♻️ Reset agent ")
                info_btn  = gr.Button("ℹ️ Agent info")


    # ===== Handlers =====

    def on_backend_change(b):
        # Update model display and show/hide tools warning
        model = MODEL_BY_BACKEND.get(b, "<unknown>")
        warn_needed = not TOOLS_CAPABLE.get(b, True)
        warn_text = (
            "⚠️ **Tools are disabled or unreliable for this backend/model.** "
            "The app will pre-gather context (file tree, searches) and the LLM will work from that."
        )
        return (
            f"**Model:** `{model}`",
            gr.update(visible=warn_needed, value=warn_text),
        )
    backend.change(on_backend_change, inputs=[backend], outputs=[model_md, tool_warn])

    def on_apply(b):
        # Print a quick summary (and you can probe the model if desired)
        model = MODEL_BY_BACKEND.get(b, "<unknown>")
        summary = f"Settings applied. Backend={b}, Model={model}, Root={T.get_action_root()}"
        return summary
    apply_btn.click(on_apply, inputs=[backend], outputs=[status_md])

    def on_explorer_pick(selected):
        root = T.set_action_root_from_any(selected)
        return root, root, f"✅ Action folder set to: {root}"
    explorer.change(on_explorer_pick, inputs=[explorer], outputs=[action_root_out, folder_input, status_md])

    def on_set_folder(path_str):
        try:
            root = T.set_action_root_from_text(path_str)
            return root, f"✅ Action folder set to: {root}"
        except Exception as e:
            return T.get_action_root(), f"❌ {e}"
    set_folder_btn.click(on_set_folder, inputs=[folder_input], outputs=[action_root_out, status_md])

    def _safe(val: str) -> str:
        # compact helper to avoid None
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    def _gather_context_for_explore() -> str:
        try:
            tree   = _safe(T.list_dir(".", "**/*"))
            counts = _safe(T.count_files_by_type(".", "**/*"))
            hits   = _safe(T.search_text(".", "**/*", r"TODO|FIXME", 200))
            return (
                "----- FILE TREE -----\n" + tree + "\n\n"
                "----- COUNTS -----\n" + counts + "\n\n"
                "----- MATCHES (TODO/FIXME) -----\n" + hits + "\n"
            )
        except Exception as e:
            return f"(context collection failed: {e})"

    def preset_explore(backend_val: str) -> str:
        if TOOLS_CAPABLE.get(backend_val, True):
            # Let the agent call tools
            return (
                "Explore this folder (and all subfolders) and produce a short, useful overview.\n"
                "Goals (no full file tree):\n"
                "1) Total number of files (recursively).\n"
                "2) File counts by extension — show .json, .py, .csv, .md explicitly and group the rest as “Other”.\n"
                "3) Find lines containing TODO or FIXME and list them as: relative/path:line: snippet (cap ~200).\n"
                "4) Write a concise markdown report with sections:\n"
                "# Folder Overview\n"
                "- Action folder: …\n"
                "- Total files: …\n"
                "## By extension\n"
                "| Ext | Count |\n"
                "## TODO / FIXME\n"
                "<file:line snippet list or “(none found)”>\n"
                "## Next Actions (3)\n"
                "<Actionable, specific items tied to findings>\n"
                "Finally, print the report and save the report as `reports/folder_overview.md'create file and folder if don't exist\n"
            )
        # Prep-first (no tools)
        ctx = _gather_context_for_explore()
        return (
            "Explore the folder using the context below (no tool calls available).\n"
            "Write a concise overview and 3 next actions.\n"
            "Save to `reports/folder_overview.md`.\n\n" + ctx
        )

    def preset_analyze(backend_val: str) -> str:
        if TOOLS_CAPABLE.get(backend_val, True):
            return (
                "Analyze documents here:\n"
                "- Read key files (txt/md/pdf/docx/csv/xlsx)\n"
                "- Summarize: Context, Highlights, Risks, Open Questions\n"
                "Save to `reports/analysis_brief.md`."
            )
        # Prep-first: we only include the tree; the model will ask if it needs specifics
        ctx = _gather_context_for_explore()
        return (
            "Analyze this folder (no tool calls available).\n"
            "Produce a brief with: Context, Highlights, Risks, Open Questions.\n"
            "Save to `reports/analysis_brief.md`.\n\n" + ctx
        )

    def preset_translate(backend_val: str) -> str:
        return (
            "Translate a document (preserve headings/lists).\n"
            "Read the document: \n"
            "Translate the returned Markdown to (insert language), preserve structure \n"
            "write the result to the file e.g., 'docs/translated.md'  \n"
        )

    def preset_extract(backend_val: str) -> str:
        return (
            "Extract structured data from semi-structured docs.\n"
            "Reply by asking me for: (1) file/glob, (2) desired fields/schema, "
            "(3) output type (CSV/JSON). Then perform extraction and save under "
            "`reports/extract.(csv|json)`."
        )

    def preset_generate(backend_val: str) -> str:
        if TOOLS_CAPABLE.get(backend_val, True):
            return (
                "Create a friendly `README.md`:\n"
                "- What’s in this folder\n- How to run\n- 3 quickstart steps\n"
                "Preview first; then write `README.md`."
            )
        ctx = _gather_context_for_explore()
        return (
            "Create a friendly `README.md` (no tool calls available).\n"
            "Use the context below.\n"
            "Preview first; then write `README.md`.\n\n" + ctx
        )

    def preset_batch(backend_val: str, safe_mode_val: bool) -> str:
        warn = ""
        if safe_mode_val:
            warn = ("⚠️ Safe Mode is ON (no shell). Turn it OFF to execute commands.\n\n")
        return (
            f"{warn}"
            "Batch actions (plan → confirm → run):\n"
            "- Propose shell commands for the requested operation (e.g., rename, ffmpeg, git)\n"
            "- WAIT for my OK\n"
            "- Then execute and log to `reports/batch_log.md`."
        )

    def on_reset_agent(backend_val, safe_mode_val):
        status = reset_agent_cache(backend_val, None, safe_mode_val)  # model_id=None → auto-resolve
        # Also clear visible chat so UI matches backend state
        model_name = MODEL_BY_BACKEND.get(backend_val, "<auto>")
        return [], f"🔄 {status} for {backend_val} / {model_name}"

    def on_show_agent_info(backend_val, safe_mode_val):
        meta = current_agent_meta(backend_val, None, safe_mode_val)
        if not meta:
            return "No agent in cache for this backend/safe-mode."
        return (
            f"**Agent** `{meta['id']}`  \n"
            f"- Backend: {meta['backend']}  \n"
            f"- Model: {meta['model']}  \n"
            f"- Safe mode: {meta['safe_mode']}  \n"
            f"- Created: {meta['created']}  \n"
            f"- Steps: {meta['steps']}"
        )

    # Presets: fill the message box with a task prompt

    btn_explore.click(lambda b: preset_explore(b), inputs=[backend], outputs=[msg])
    btn_analyze.click(lambda b: preset_analyze(b), inputs=[backend], outputs=[msg])
    btn_translate.click(lambda b: preset_translate(b), inputs=[backend], outputs=[msg])
    btn_extract.click(lambda b: preset_extract(b), inputs=[backend], outputs=[msg])
    btn_generate.click(lambda b: preset_generate(b), inputs=[backend], outputs=[msg])
    btn_batch.click(lambda b, s: preset_batch(b, s), inputs=[backend, safe_mode], outputs=[msg])
    reset_btn.click(on_reset_agent, inputs=[backend, safe_mode], outputs=[chatbot, status_md])
    info_btn.click(on_show_agent_info, inputs=[backend, safe_mode], outputs=[status_md])

    # Chat submit/send
    def respond(message, history, backend_val, safe_mode_val, show_logs_val, max_steps_val):
        # Model id is auto-resolved in agent_core; pass None/"" here
        text, _ = run_agent_message(message, backend_val, None, safe_mode_val, show_logs_val, max_steps_val)
        hist = history or []
        hist += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text},
        ]
        return hist, ""  # clear input box

    send_btn.click(respond, inputs=[msg, chatbot, backend, safe_mode, show_logs, max_steps], outputs=[chatbot, msg])
    msg.submit(respond, inputs=[msg, chatbot, backend, safe_mode, show_logs, max_steps], outputs=[chatbot, msg])

    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
