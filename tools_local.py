"""tools_local.py
Local tool implementations.

All custom tools and sandbox utilities (e.g., file/O, OS helpers)
are defined here, following the MCP‑style separation of concerns.
"""


"""
All tools live here, separate from the UI.
They are sandboxed to ACTION_ROOT. The UI sets ACTION_ROOT via setters below.

Design:
- ACTION_ROOT: global sandbox folder the agent can touch
- LAST_EXPLORER_PICK: last item clicked in FileExplorer (file or folder)
- get_tools(safe_mode): returns list of @tool functions (no shell in safe mode)
"""


from pathlib import Path
from typing import Union, List, Optional
import os, re, io
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from smolagents import tool

# --------- Global sandbox state ----------
ACTION_ROOT: Path = Path.cwd().resolve()
LAST_EXPLORER_PICK: Optional[Path] = None

# OCR / PDF helpers (env-configurable in app startup)
DEFAULT_OCR_LANGS = os.getenv("OCR_LANGS", "eng")
POPPLER_PATH = os.getenv("POPPLER_PATH")  # for pdf2image on Windows
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --------- Folder setters / helpers ----------
def _extract_path_from_explorer_selection(selected) -> Optional[Path]:
    """Robustly parse FileExplorer selection across Gradio versions."""
    if not selected:
        return None

    def _one(x):
        if x is None:
            return None
        if isinstance(x, (str, Path)):
            p = Path(x)
            return p if p.is_absolute() else (Path.cwd() / p)
        if isinstance(x, dict):
            # Common shapes: {"path": "..."} or {"name": "..."}
            if x.get("path"):
                p = Path(x["path"])
                return p if p.is_absolute() else (Path.cwd() / p)
            if x.get("name"):
                return (Path.cwd() / str(x["name"]))
        return None

    if isinstance(selected, list) and selected:
        return _one(selected[0])
    return _one(selected)

def set_action_root_from_any(selected) -> str:
    """Set ACTION_ROOT from FileExplorer selection (file → parent)."""
    global ACTION_ROOT, LAST_EXPLORER_PICK
    p = _extract_path_from_explorer_selection(selected)
    if p is None:
        return str(ACTION_ROOT)
    LAST_EXPLORER_PICK = p
    ACTION_ROOT = p if p.is_dir() else p.parent
    return str(ACTION_ROOT)

def set_action_root_from_text(path_str: str) -> str:
    """Set ACTION_ROOT from a textbox path."""
    global ACTION_ROOT
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"path not found: {p}")
    ACTION_ROOT = p if p.is_dir() else p.parent
    return str(ACTION_ROOT)

def get_action_root() -> str:
    return str(ACTION_ROOT)

def _normalize_to_root(user_path: str) -> Path:
    """Resolve a path (absolute or relative) under ACTION_ROOT; block escapes."""
    p = Path(user_path)
    rp = (ACTION_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if ACTION_ROOT == rp or ACTION_ROOT in rp.parents:
        return rp
    raise ValueError(f"Path escapes sandbox: {rp} not under {ACTION_ROOT}")

# --------- General tools ----------
@tool
def cwd() -> str:
    """Return the absolute path of the current action folder.

    Returns:
        str: Absolute path to ACTION_ROOT.
    """
    return str(ACTION_ROOT)

@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file (sandboxed to ACTION_ROOT).

    Args:
        path (str): Relative path inside the action folder where the file is written.
        content (str): Text content to write.

    Returns:
        str: 'saved:<relative-path>' on success or 'error:<message>' on failure.
    """
    try:
        target = _normalize_to_root(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"saved:{target.relative_to(ACTION_ROOT)}"
    except Exception as e:
        return f"error:{e}"

@tool
def read_text(path: str, max_chars: int = 20000) -> str:
    """Read a text file from the action folder with robust decoding.

    Args:
        path (str): Relative/absolute file path under action folder.
        max_chars (int): Max characters to return.

    Returns:
        str: File contents (possibly truncated) or 'error:<message>'.
    """
    try:
        target = _normalize_to_root(path)
        data = target.read_bytes()
        for enc in ("utf-8", "cp1251", "cp866", "utf-16", "latin-1"):
            try:
                return data.decode(enc)[:max_chars]
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")[:max_chars]
    except Exception as e:
        return f"error:{e}"

@tool
def list_dir(path: str = ".", glob_pattern: str = "*") -> str:
    """List files/folders that match a glob pattern.

    Args:
        path (str): Subfolder under the action folder.
        glob_pattern (str): Glob to match (e.g., '*.md', '**/*.py').

    Returns:
        str: Matches relative to action folder, or '(no matches)'.
    """
    try:
        base = _normalize_to_root(path)
        items = [str(p.relative_to(ACTION_ROOT)) for p in base.glob(glob_pattern)]
        return "\n".join(items) if items else "(no matches)"
    except Exception as e:
        return f"error:{e}"

@tool
def search_text(path: str = ".", glob_pattern: str = "**/*", query: str = "", max_hits: int = 100) -> str:
    """Case-insensitive regex search within files.

    Args:
        path (str): Subfolder under the action folder.
        glob_pattern (str): File glob to include (e.g., '**/*.py').
        query (str): Regular expression (case-insensitive).
        max_hits (int): Max number of matched lines.

    Returns:
        str: '<relative-path>:<line-no>: <line>' or '(no hits)'.
    """
    try:
        rx = re.compile(query, re.IGNORECASE)
        root = _normalize_to_root(path)
        hits = []
        for f in root.rglob(glob_pattern):
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(ACTION_ROOT)}:{i}: {line.strip()}")
                        if len(hits) >= max_hits:
                            return "\n".join(hits)
            except Exception:
                continue
        return "\n".join(hits) if hits else "(no hits)"
    except Exception as e:
        return f"error:{e}"

@tool
def sh(cmd: str, shell: str = "auto") -> str:
    """Execute a shell command inside the action folder with UTF-8 output.

    Args:
        cmd (str): Command to execute (Windows: 'dir', 'type', 'findstr').
        shell (str): 'auto'|'cmd'|'powershell' (Windows only).

    Returns:
        str: stdout+stderr (UTF-8) or 'error:<message>'.
    """
    try:
        import subprocess, os as _os
        cmd_lower = cmd.strip().lower()
        if cmd_lower in ("pwd", "cd", "cwd"):
            return str(ACTION_ROOT)
        if cmd_lower in ("printenv", "env"):
            keys = ["USERNAME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERDOMAIN", "COMPUTERNAME", "PATH"]
            return "\n".join(f"{k}={_os.environ.get(k, '')}" for k in keys)

        if _os.name == "nt":
            if shell in ("auto", "cmd"):
                cmd_win = f"chcp 65001 >NUL & {cmd}"
                r = subprocess.run(
                    cmd_win, shell=True, cwd=str(ACTION_ROOT),
                    capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                return (r.stdout or "") + (r.stderr or "")
            else:
                ps_cmd = f"[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); {cmd}"
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    cwd=str(ACTION_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                return (r.stdout or "") + (r.stderr or "")
        else:
            r = subprocess.run(
                cmd, shell=True, cwd=str(ACTION_ROOT),
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"error:{e}"

# --------- Document formats ----------
@tool
def read_pdf(path: str, max_pages: int = 20) -> str:
    """Extract text from a non-scanned PDF (no OCR).

    Args:
        path (str): PDF path under action folder.
        max_pages (int): Max pages to extract.

    Returns:
        str: Text or 'error:<message>'.
    """
    try:
        from pypdf import PdfReader
        target = _normalize_to_root(path)
        reader = PdfReader(str(target))
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            pages.append(page.extract_text() or "")
        txt = "\n\n".join(pages).strip()
        return txt if txt else "(no extractable text)"
    except Exception as e:
        return f"error:{e}"

@tool
def read_pdf_ocr(path: str, max_pages: int = 10, dpi: int = 300, lang: str = "", preprocess: str = "auto", max_chars: int = 20000) -> str:
    """OCR a scanned PDF by rendering pages to images then running Tesseract.

    Args:
        path (str): PDF path under action folder.
        max_pages (int): Max pages from start.
        dpi (int): Render DPI for OCR (300–400 for quality).
        lang (str): Tesseract languages (e.g., 'eng+rus').
        preprocess (str): 'auto'|'none'|'strong'.
        max_chars (int): Truncate output length.

    Returns:
        str: Extracted text (maybe truncated) or 'error:<message>'.
    """
    try:
        from pdf2image import convert_from_path
        target = _normalize_to_root(path)
        kwargs = {"dpi": dpi}
        if POPPLER_PATH: kwargs["poppler_path"] = POPPLER_PATH
        images = convert_from_path(str(target), **kwargs)
        out, langs = [], (lang or DEFAULT_OCR_LANGS).strip() or "eng"
        for i, img in enumerate(images, 1):
            if i > max_pages: break
            page = img
            if preprocess != "none":
                page = ImageOps.grayscale(page)
                if preprocess in ("auto", "strong"):
                    page = ImageOps.autocontrast(page).filter(ImageFilter.MedianFilter(size=3))
                if preprocess == "strong":
                    page = page.point(lambda p: 255 if p > 160 else 0)
            text = pytesseract.image_to_string(page, lang=langs).strip()
            out.append(f"[page {i}]\n{text}")
        txt = "\n\n".join(out).strip()
        return txt[:max_chars] if txt else "(no text found)"
    except Exception as e:
        return f"error:{e}"

@tool
def read_image_ocr(path: str, lang: str = "", preprocess: str = "auto", max_chars: int = 20000) -> str:
    """OCR any image using Tesseract.

    Args:
        path (str): Image path under action folder.
        lang (str): Tesseract languages (e.g., 'eng+rus').
        preprocess (str): 'auto'|'none'|'strong'.
        max_chars (int): Truncate output.

    Returns:
        str: Extracted text or 'error:<message>'.
    """
    try:
        target = _normalize_to_root(path)
        img = Image.open(str(target))
        if preprocess != "none":
            img = ImageOps.grayscale(img)
            if preprocess in ("auto", "strong"):
                img = ImageOps.autocontrast(img).filter(ImageFilter.MedianFilter(size=3))
            if preprocess == "strong":
                img = img.point(lambda p: 255 if p > 160 else 0)
        langs = (lang or DEFAULT_OCR_LANGS).strip() or "eng"
        text = pytesseract.image_to_string(img, lang=langs)
        return (text or "").strip()[:max_chars] or "(no text found)"
    except Exception as e:
        return f"error:{e}"

@tool
def read_docx(path: str, max_chars: int = 20000) -> str:
    """Extract text from a DOCX file.

    Args:
        path (str): DOCX path under action folder.
        max_chars (int): Truncate length.

    Returns:
        str: Text or 'error:<message>'.
    """
    try:
        import docx
        target = _normalize_to_root(path)
        doc = docx.Document(str(target))
        text = "\n".join(p.text for p in doc.paragraphs) or ""
        return text[:max_chars] if text else "(empty docx)"
    except Exception as e:
        return f"error:{e}"

@tool
def read_xlsx(path: str, sheet: str = None, max_rows: int = 50) -> str:
    """Read a few rows from an XLSX sheet.

    Args:
        path (str): XLSX path under action folder.
        sheet (str): Sheet name (None=first).
        max_rows (int): Max rows.

    Returns:
        str: Simple preview or 'error:<message>'.
    """
    try:
        from openpyxl import load_workbook
        target = _normalize_to_root(path)
        wb = load_workbook(str(target), read_only=True, data_only=True)
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb[wb.sheetnames[0]]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            if i > max_rows: break
            rows.append(", ".join("" if v is None else str(v) for v in row))
        wb.close()
        return "\n".join(rows) if rows else "(empty sheet)"
    except Exception as e:
        return f"error:{e}"

@tool
def read_csv(path: str, delimiter: str = "", max_rows: int = 100) -> str:
    """Preview a CSV/TSV with delimiter detection.

    Args:
        path (str): CSV path under action folder.
        delimiter (str): Explicit delimiter (optional).
        max_rows (int): Max rows.

    Returns:
        str: Header + rows or 'error:<message>'.
    """
    try:
        import csv
        target = _normalize_to_root(path)
        data = target.read_bytes()
        text = None
        for enc in ("utf-8", "cp1251", "utf-16", "latin-1"):
            try:
                text = data.decode(enc); break
            except Exception: continue
        if text is None: text = data.decode("utf-8", errors="replace")
        sniffer = csv.Sniffer()
        if delimiter:
            dialect = csv.excel
            dialect.delimiter = delimiter  # type: ignore[attr-defined]
        else:
            try:
                dialect = sniffer.sniff(text.splitlines()[0])
            except Exception:
                dialect = csv.excel
        out, reader = [], csv.reader(text.splitlines(), dialect=dialect)
        for i, row in enumerate(reader):
            if i > max_rows: break
            out.append(", ".join(row))
        return "\n".join(out) if out else "(empty csv)"
    except Exception as e:
        return f"error:{e}"

from collections import Counter
@tool
def count_files_by_type(path: str = ".", glob_pattern: str = "**/*") -> str:
    """Count files grouped by extension under the action folder.

    Args:
        path (str): Subfolder under the action folder.
        glob_pattern (str): Glob to include (e.g., '**/*').

    Returns:
        str: JSON dict mapping extension (like '.py') to counts, '' for no extension.
    """
    try:
        import json
        base = _normalize_to_root(path)
        cnt = Counter()
        for p in base.rglob("*"):
            if p.is_file():
                cnt[p.suffix or ""] += 1
        return json.dumps(dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"error:{e}"


# --------- Tool registry ----------
def get_tools(safe_mode: bool):
    base = [write_file, read_text, list_dir, search_text, cwd,
            read_pdf, read_pdf_ocr, read_image_ocr, read_docx, read_xlsx, read_csv,
            count_files_by_type]
    return base if safe_mode else base + [sh]
