import os
import asyncio
import logging
import sys
import json
import base64
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import io
import re

# FastAPI
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Path, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Core Processing & External Libraries
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai
from qdrant_client import QdrantClient, models
from PIL import Image
import httpx
import aiofiles
from urllib.parse import urljoin
import aiohttp
from unstructured.partition.pdf import partition_pdf
from markdown_it import MarkdownIt

# Optional: PyMuPDF for fast page counting
try:
    import fitz
    _fitz_available = True
except ImportError:
    _fitz_available = False
    logging.warning("PyMuPDF (fitz) not installed. Page counting might be slower.")

# Optional: pypdf as another fast fallback
try:
    import pypdf
    _pypdf_available = True
except ImportError:
    _pypdf_available = False

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
dotenv_path_key = os.path.join(os.path.dirname(__file__), 'key.env')
if os.path.exists(dotenv_path_key):
    load_dotenv(dotenv_path=dotenv_path_key)
    logger.info("โหลดตัวแปรสภาพแวดล้อมจาก key.env แล้ว")

# API Keys and URLs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
LIBRARY_API_TOKEN = os.getenv("LIBRARY_API_TOKEN")
LIBRARY_API_BASE_URL = os.getenv("LIBRARY_API_BASE_URL")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# --- Global Settings ---
CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
PAGE_CONCURRENCY = int(os.getenv("PAGE_CONCURRENCY", "2"))
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))

OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME")
FALLBACK_EMBEDDING_URL = os.getenv("FALLBACK_EMBEDDING_URL")
FALLBACK_EMBEDDING_MODEL_NAME = os.getenv("FALLBACK_EMBEDDING_MODEL_NAME")

OCR_LANGUAGES = [s.strip() for s in os.getenv("OCR_LANGUAGES", "eng").split(",") if s.strip()]
INFER_TABLE_STRUCTURE = os.getenv("INFER_TABLE_STRUCTURE", "true").lower() in ("1","true","yes")

# Markdown export
OCR_MD_OUTPUT_DIR = os.getenv("OCR_MD_OUTPUT_DIR", "ocr_md_outputs")
SAVE_MD_ENABLED = os.getenv("SAVE_MD_ENABLED", "true").lower() in ("1", "true", "yes")
os.makedirs(OCR_MD_OUTPUT_DIR, exist_ok=True)

# Rebuild Markdown for whole document with structure heuristics
MD_REBUILD_WHOLEDOC = os.getenv("MD_REBUILD_WHOLEDOC", "true").lower() in ("1", "true", "yes")
MD_CLI_HINTS = [s.strip() for s in os.getenv(
    "MD_CLI_HINTS",
    "kubectl,docker,git,pip,npm,helm,aws,az,gcloud,psql,redis-cli,curl,wget,python,node,go,java,php,perl,ruby,bash,sh,powershell,terraform,ansible,ffmpeg,ffprobe,scp,ssh,rsync,sqlcmd,psql,mongo,mysql"
).split(",") if s.strip()]

# --- Large PDF / Batch Mode (เหมาะกับสแกน/รูป/ตารางเยอะ) ---
LARGE_PDF_MODE = os.getenv("LARGE_PDF_MODE", "true").lower() in ("1", "true", "yes")
BATCH_PAGE_SIZE = int(os.getenv("BATCH_PAGE_SIZE", "30"))
PREVIEW_TABLES = os.getenv("PREVIEW_TABLES", "true").lower() in ("1","true","yes")

# --- Gemini for Markdown post-process (optional) ---
USE_GEMINI_MD = os.getenv("USE_GEMINI_MD", "false").lower() in ("1","true","yes")
GEMINI_MD_MODEL = os.getenv("GEMINI_MD_MODEL", "gemini-2.0-flash")
GEMINI_MD_TEMPERATURE = float(os.getenv("GEMINI_MD_TEMPERATURE", "0.0"))
GEMINI_MD_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MD_MAX_OUTPUT_TOKENS", "8192"))

# Toggle for image caption cost control
USE_GEMINI_CAPTION = os.getenv("USE_GEMINI_CAPTION", "true").lower() in ("1","true","yes")

JOB_STATUS_FILE = os.path.join(OCR_MD_OUTPUT_DIR, "job_status.json")
job_status_lock = asyncio.Lock()

# --- Idle Monitor Settings ---
IDLE_THRESHOLD_SECONDS = int(os.getenv("IDLE_THRESHOLD_SECONDS", "1800"))
IDLE_CHECK_INTERVAL_SECONDS = int(os.getenv("IDLE_CHECK_INTERVAL_SECONDS", "60"))
LAST_ACTIVITY_AT = datetime.now(timezone.utc)
last_activity_lock = asyncio.Lock()
idle_notified = False
idle_watchdog_task: Optional[asyncio.Task] = None

# --- Gemini Prompts ---
IMAGE_CAPTION_PROMPT = "Describe this image for a search index. What key information does it contain? Be concise and informative, focusing on data and relationships shown."

# --- Global Client Instances ---
vision_model: Optional[genai.GenerativeModel] = None
md_model: Optional[genai.GenerativeModel] = None
qdrant_client: Optional[QdrantClient] = None
semaphore_embedding_call: Optional[asyncio.Semaphore] = None
page_processing_semaphore: Optional[asyncio.Semaphore] = None
embedding_http_session: Optional[aiohttp.ClientSession] = None

# --- Pydantic Models ---
class ProcessingResponse(BaseModel):
    collection_name: str
    status: str
    processed_chunks: int
    failed_chunks: int
    message: str
    file_name: str

class LibrarySearchRequest(BaseModel):
    query: str
    pages: Optional[str] = None
    collection_name: Optional[str] = None

class ProcessByPathRequest(BaseModel):
    file_path: str
    pages: Optional[str] = None
    collection_name: Optional[str] = None

class SourceListResponse(BaseModel):
    collection_name: str
    source_count: int
    sources: List[str]

class AcknowledgementResponse(BaseModel):
    message: str
    file_path: str
    task_status: str

class JobStatus(BaseModel):
    status: str
    file_path: str
    created_at: str
    updated_at: str
    result: Optional[ProcessingResponse] = None
    error: Optional[str] = None

class JobStatusResponse(BaseModel):
    file_path: str
    details: JobStatus

class CleanupByBookIdRequest(BaseModel):
    book_id: str
    collection_name: Optional[str] = None
    dry_run: bool = False

class CleanupByBookIdResponse(BaseModel):
    total_matched: int
    total_deleted: int
    dry_run: bool
    details: List[Dict[str, Any]]

# ---------- Helpers ----------
def _safe_filename(name: str) -> str:
    keep = "-_.() []"
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in keep).strip()
    return cleaned or f"doc_{uuid.uuid4().hex}"

def _chunk_list(items: List[int], n: int) -> List[List[int]]:
    return [items[i:i+n] for i in range(0, len(items), n)]

# ---------- Activity & Idle ----------
async def mark_activity(source: str = ""):
    global LAST_ACTIVITY_AT, idle_notified
    async with last_activity_lock:
        LAST_ACTIVITY_AT = datetime.now(timezone.utc)
        idle_notified = False

async def notify_startup():
    if not N8N_WEBHOOK_URL:
        logger.warning("[Startup] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    payload = {
        "status": "online",
        "message": "PDF Processing Service has started successfully.",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(N8N_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
            logger.info("ส่ง Startup Webhook notification สำเร็จ")
    except Exception as e:
        logger.error(f"ส่ง Startup Webhook ล้มเหลว: {e}")

async def notify_idle(idle_seconds: int):
    if not N8N_WEBHOOK_URL:
        return
    payload = {
        "status": "idle",
        "message": f"Service has been idle for {idle_seconds} seconds.",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(N8N_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
            logger.info("ส่ง Idle Webhook notification สำเร็จ")
    except Exception as e:
        logger.error(f"ส่ง Idle Webhook ล้มเหลว: {e}")

async def idle_watchdog():
    global idle_notified
    logger.info(f"[Idle] Idle watchdog started (threshold={IDLE_THRESHOLD_SECONDS}s)")
    while True:
        try:
            await asyncio.sleep(IDLE_CHECK_INTERVAL_SECONDS)
            now = datetime.now(timezone.utc)
            async with last_activity_lock:
                last_ts = LAST_ACTIVITY_AT
            idle_secs = int((now - last_ts).total_seconds())
            if idle_secs >= IDLE_THRESHOLD_SECONDS and not idle_notified:
                await notify_idle(idle_secs)
                idle_notified = True
                logger.info(f"[Idle] Notified idle webhook for {idle_secs}s")
        except asyncio.CancelledError:
            logger.info("[Idle] Idle watchdog canceled")
            break
        except Exception as e:
            logger.error(f"[Idle] Idle watchdog error: {e}", exc_info=True)

# ---------- Unstructured helpers ----------
def parse_markdown_table(md_text: str) -> list[dict]:
    if not md_text or not isinstance(md_text, str):
        return []
    try:
        md = MarkdownIt()
        tokens = md.parse(md_text)
        header, rows, in_tbody = [], [], False
        for i, token in enumerate(tokens):
            if token.type == 'th_open' and (i + 1 < len(tokens)) and tokens[i+1].type == 'inline':
                header.append(tokens[i+1].content)
            if token.type == 'tbody_open':
                in_tbody = True
            if token.type == 'tr_open' and in_tbody:
                close_tr_index = -1
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == 'tr_close':
                        close_tr_index = j
                        break
                if close_tr_index == -1:
                    continue
                row_tokens = tokens[i+1: close_tr_index]
                cell_contents = [t.content for t in row_tokens if t.type == 'inline']
                current_row = {h: cell for h, cell in zip(header, cell_contents)}
                if current_row:
                    rows.append(current_row)
        return rows
    except Exception as e:
        logger.error(f"Error parsing markdown table: {e}")
        return []

async def generate_image_caption(image: Image.Image) -> str:
    global vision_model
    if not USE_GEMINI_CAPTION or not vision_model:
        return "No description available."
    try:
        prompt = [IMAGE_CAPTION_PROMPT, image]
        response = await asyncio.to_thread(vision_model.generate_content, prompt)
        return response.text or "Image could not be described."
    except Exception as e:
        logger.error(f"Error generating image caption: {e}")
        return "Error in image description generation."

def _elem_page_number(el) -> Optional[int]:
    md = getattr(el, "metadata", None)
    if md is None:
        return None
    pn = getattr(md, "page_number", None)
    if pn is None and isinstance(md, dict):
        pn = md.get("page_number")
    return pn

def make_chunk_id(meta: Dict[str, Any], text: Optional[str] = None) -> str:
    base = {
        "source": meta.get("source"),
        "page": ((meta.get("loc") or {}).get("pageNumber")),
        "elementIndex": ((meta.get("loc") or {}).get("elementIndex")),
        "content_type": meta.get("content_type"),
        "table_id": meta.get("table_id"),
        "row_index": meta.get("row_index"),
        "book_id": meta.get("book_id"),
    }
    if text:
        base["text_hash"] = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    seed = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

# -------- Text cleanup and noise --------
def fix_joined_words(t: str) -> str:
    if not t:
        return t
    t = re.sub(r'([a-z0-9\)\]])([A-Z])', r'\1 \2', t)
    t = re.sub(r':(?!\s)', ': ', t)
    t = re.sub(r'\s+([,.;:])', r'\1', t)
    return t

def dehyphenate_wraps(text: str) -> str:
    return re.sub(r'(?<=\w)-\s+(?=\w)', '', text or "")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    t = dehyphenate_wraps(t)
    t = fix_joined_words(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t).strip()
    return t

def is_noise(el) -> bool:
    cat = getattr(el, "category", "")
    txt = (getattr(el, "text", "") or "").strip()
    if not txt:
        return True
    if cat in {"UncategorizedText", "PageBreak"}:
        return True
    alnum_ratio = sum(ch.isalnum() for ch in txt) / max(len(txt), 1)
    if alnum_ratio < 0.15:
        return True
    return False

# -------- Coordinates & reading order --------
def _coord_values(el) -> Optional[Dict[str, float]]:
    try:
        md = el.metadata.to_dict() if getattr(el, "metadata", None) else {}
        coords = md.get("coordinates") or {}
        points = coords.get("points") or []
        xs, ys = [], []
        for p in points:
            if isinstance(p, dict):
                xs.append(float(p.get("x", 0.0)))
                ys.append(float(p.get("y", 0.0)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
        if not xs or not ys:
            return None
        layout_w = float(coords.get("layout_width") or 0.0)
        layout_h = float(coords.get("layout_height") or 0.0)
        info = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "W": layout_w, "H": layout_h}
        if layout_w > 0: info["nx"] = info["min_x"] / layout_w
        if layout_h > 0:
            info["ny_top"] = info["min_y"] / layout_h
            info["ny_bot"] = info["max_y"] / layout_h
        return info
    except Exception:
        return None

def sort_by_reading_order(elements: List[Any]) -> List[Any]:
    def key(el):
        pn = _elem_page_number(el) or 1
        cv = _coord_values(el) or {}
        y = cv.get("ny_top", cv.get("min_y", 1e9))
        x = cv.get("nx", cv.get("min_x", 0.0))
        return (pn, y, x)
    return sorted(elements, key=key)

HEADER_NOISE_PATTERNS = [
    r"Stansberry Research",
    r"\bManaging Editor\b",
    r"\bIN THIS ISSUE\b",
    r"\bNEXT ISSUE\b",
    r"\bwww\.[A-Za-z0-9\-\.]+",
    r"\bBaltimore\b",
    r"\bSaint Paul Street\b",
]

def matches_noise_pattern(text: str) -> bool:
    for pat in HEADER_NOISE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def is_header_footer(el, header_thresh: float = 0.06, footer_thresh: float = 0.92) -> bool:
    cv = _coord_values(el) or {}
    ny_top = cv.get("ny_top", None)
    ny_bot = cv.get("ny_bot", None)
    txt = clean_text(getattr(el, "text", "") or "")
    if matches_noise_pattern(txt):
        return True
    if ny_top is None or ny_bot is None:
        return False
    if ny_top <= header_thresh or ny_bot >= footer_thresh:
        return True
    return False

# -------- Code/JSON detection --------
def token_ratio(s: str, chars: str = r'\-\[\]\{\}\(\)=_/\\\|\$><;,:') -> float:
    if not s: return 0.0
    hits = sum(1 for ch in s if re.search(f"[{re.escape(chars)}]", ch))
    return hits / len(s)

def looks_jsonish(s: str) -> bool:
    if not s: return False
    t = s.strip()
    if not t: return False
    if t[0] in "{[":
        try:
            json.loads(t)
            return True
        except Exception:
            return (t.count(":") >= 2) and any(ch in t for ch in "{}[]")
    return (t.count(":") >= 2) and any(ch in t for ch in "{}[]")

def guess_code_lang(s: str) -> str:
    t = s.strip()
    if looks_jsonish(t): return "json"
    if re.search(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b', t, re.I): return "sql"
    if re.search(r'\b(def|import|from|class|print\()', t): return "python"
    if re.search(r'\b(const|let|var|function)\b|\bconsole\.log\(', t): return "javascript"
    if re.search(r'\bpackage\s+main\b|\bfunc\s+\w+\(', t): return "go"
    if re.search(r'^\s*(curl|wget|sudo|export|echo|cd|ls|cat|grep|awk|sed|tar|zip)\b', t): return "bash"
    if re.search(r'^\s*(kubectl|docker|helm|aws|az|gcloud|git|pip|npm|node|python|java|php|perl|ruby|psql|redis-cli)\b', t, re.I): return "bash"
    if re.search(r'^\s*Set-Item|Get-Item|Write-Host|Get-ChildItem\b', t): return "powershell"
    return ""

def is_cli_command_line(s: str) -> bool:
    if not s: return False
    t = s.strip()
    if re.search(r'\s--?\w+', t) or re.search(r'[/\\][\w\-/\.]+', t) or re.search(r'[|><]{1,2}', t):
        return True
    first = t.split()[0] if t.split() else ""
    if first.lower() in [h.lower() for h in MD_CLI_HINTS]:
        return True
    if re.search(r'[\[\{].+[\]\}]', t) and "=" in t:
        return True
    if re.search(r'\s\d+$', t):
        return True
    if token_ratio(t) > 0.12 and len(t) <= 180:
        return True
    return False

def is_commandish_title(txt: str) -> bool:
    s = txt.strip()
    if not s: return False
    if (("[" in s and "]" in s) or ("(" in s and ")" in s)) and (re.search(r'--?\w+', s) or "=" in s):
        return True
    if re.match(r'^[a-z0-9][\w\-]*(\s+[\w\-/\[\]\(\)=\.:]+)+$', s):
        return True
    return False

def fence(content: str, lang: str = "") -> str:
    content = (content or "").strip("\n")
    return f"```{lang}\n{content}\n```" if lang else f"```\n{content}\n```"

def make_anchor(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\- ]+", "", s).strip().lower()
    s = s.replace(" ", "-")
    return s or f"section-{uuid.uuid4().hex[:8]}"

def detect_document_title(elements: List[Any], fallback: str) -> str:
    candidates = []
    for el in elements:
        if getattr(el, "category", "") == "Title" and (_elem_page_number(el) or 1) == 1:
            txt = clean_text(getattr(el, "text", "") or "")
            if not txt: continue
            cv = _coord_values(el) or {}
            ny_top = cv.get("ny_top", 1.0)
            upper_ratio = (sum(ch.isupper() for ch in txt if ch.isalpha()) / max(1, sum(ch.isalpha() for ch in txt)))
            score = (1.0 - min(ny_top, 1.0)) + upper_ratio + min(len(txt), 120) / 120.0
            candidates.append((score, txt))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return fallback

def render_table(el) -> str:
    md = el.metadata.to_dict() if getattr(el, "metadata", None) else {}
    html = md.get("text_as_html")
    if html:
        return html.strip()
    return clean_text(getattr(el, "text", "") or "")

def render_image(el, image_output_dir: str) -> str:
    md = el.metadata.to_dict() if getattr(el, "metadata", None) else {}
    img_path = md.get("image_path")
    if not img_path:
        paths = md.get("image_paths") or []
        img_path = paths[0] if paths else None
    if img_path and os.path.exists(img_path):
        rel = os.path.relpath(img_path, OCR_MD_OUTPUT_DIR)
        return f"![]({rel})"
    if hasattr(el, "image_base64") and el.image_base64:
        try:
            raw = base64.b64decode(el.image_base64)
            os.makedirs(image_output_dir, exist_ok=True)
            fname = f"img_{uuid.uuid4().hex}.png"
            fpath = os.path.join(image_output_dir, fname)
            with open(fpath, "wb") as f:
                f.write(raw)
            rel = os.path.relpath(fpath, OCR_MD_OUTPUT_DIR)
            return f"![]({rel})"
        except Exception:
            return ""
    return ""

def group_sections_by_title(elements: List[Any], image_output_dir: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current = {"title": None, "blocks": []}

    def push_current():
        nonlocal current
        if current["title"] or current["blocks"]:
            merged = []
            code_acc = None
            for b in current["blocks"]:
                if b["type"] == "code":
                    lang = b.get("lang") or ""
                    if code_acc and code_acc["lang"] == lang:
                        code_acc["content"].append(b["content"])
                    else:
                        if code_acc:
                            code_acc["content"] = "\n".join(code_acc["content"])
                            merged.append(code_acc)
                        code_acc = {"type": "code", "lang": lang, "content": [b["content"]]}
                else:
                    if code_acc:
                        code_acc["content"] = "\n".join(code_acc["content"])
                        merged.append(code_acc); code_acc = None
                    merged.append(b)
            if code_acc:
                code_acc["content"] = "\n".join(code_acc["content"])
                merged.append(code_acc)
            current["blocks"] = merged
            sections.append(current)
        current = {"title": None, "blocks": []}

    list_buffer: List[str] = []
    def flush_list():
        nonlocal list_buffer
        if list_buffer:
            current["blocks"].append({"type": "ul", "content": list_buffer[:]})
            list_buffer = []

    for el in elements:
        if is_noise(el) or is_header_footer(el):
            continue
        cat = getattr(el, "category", "")
        raw = getattr(el, "text", "") or ""
        txt = clean_text(raw)
        if not txt:
            continue

        if cat == "Title" and is_commandish_title(txt):
            flush_list()
            current["blocks"].append({"type": "code", "lang": guess_code_lang(txt) or "bash", "content": txt})
            continue

        if cat == "Title":
            flush_list()
            push_current()
            current["title"] = txt
            continue

        if cat == "ListItem":
            list_buffer.append(txt)
            continue

        if cat == "Table":
            flush_list()
            table_md = render_table(el)
            if table_md:
                current["blocks"].append({"type": "table", "content": table_md})
            continue

        if cat in {"Image", "Figure"}:
            flush_list()
            img_md = render_image(el, image_output_dir)
            if img_md:
                current["blocks"].append({"type": "img", "content": img_md})
            continue

        flush_list()
        if looks_jsonish(txt):
            current["blocks"].append({"type": "code", "lang": "json", "content": txt})
        elif is_cli_command_line(txt) or token_ratio(txt) > 0.18:
            current["blocks"].append({"type": "code", "lang": guess_code_lang(txt) or "bash", "content": txt})
        else:
            current["blocks"].append({"type": "p", "content": txt})

    flush_list()
    push_current()
    return sections

async def rebuild_markdown_whole_doc(
    file_bytes: bytes,
    pages_to_iterate: List[int],
    source_label: str,
    image_output_dir: str,
) -> str:
    try:
        doc_elements = await asyncio.to_thread(
            partition_pdf,
            file=io.BytesIO(file_bytes),
            strategy="hi_res",
            include_metadata=True,
            infer_table_structure=INFER_TABLE_STRUCTURE,
            extract_images_in_pdf=True,
            image_output_dir_path=image_output_dir,
            languages=OCR_LANGUAGES,
            pages=pages_to_iterate,
        )
    except Exception as e:
        logger.warning(f"[MD Rebuild] partition_pdf ล้มเหลว จะ fallback per-page: {e}")
        return ""

    doc_elements = sort_by_reading_order(doc_elements)

    doc_title = detect_document_title(doc_elements, fallback=(source_label or "Document"))
    lines: List[str] = [f"# {doc_title}", ""]

    pages: Dict[int, List[Any]] = {}
    for el in doc_elements:
        pn = _elem_page_number(el) or 1
        pages.setdefault(pn, []).append(el)

    for pn in sorted(p for p in pages.keys() if p in set(pages_to_iterate)):
        page_els = [el for el in pages[pn] if not (is_noise(el) or is_header_footer(el))]
        page_els = sort_by_reading_order(page_els)

        lines.append(f"## Page {pn}")
        lines.append("")

        para_buf: List[str] = []
        list_buf: List[str] = []

        def flush_paragraph():
            nonlocal para_buf
            if para_buf:
                paragraph = clean_text(" ".join(para_buf).strip())
                if paragraph:
                    lines.append(paragraph)
                    lines.append("")
            para_buf = []

        def flush_list():
            nonlocal list_buf
            if list_buf:
                for li in list_buf:
                    lines.append(f"- {clean_text(li)}")
                lines.append("")
            list_buf = []

        for el in page_els:
            cat = getattr(el, "category", "")
            raw = getattr(el, "text", "") or ""
            txt = clean_text(raw)

            if not txt and cat not in {"Image", "Figure", "Table"}:
                continue

            if cat == "Title":
                flush_paragraph(); flush_list()
                if is_commandish_title(txt):
                    lines.append(fence(txt, guess_code_lang(txt) or "bash"))
                    lines.append("")
                else:
                    lines.append(f"## {txt}")
                    lines.append("")
                continue

            if cat == "ListItem":
                flush_paragraph()
                if txt:
                    list_buf.append(txt)
                continue

            if cat == "Table":
                flush_paragraph(); flush_list()
                tbl = render_table(el)
                if tbl:
                    lines.append(tbl); lines.append("")
                continue

            if cat in {"Image", "Figure"}:
                flush_paragraph(); flush_list()
                img_md = render_image(el, image_output_dir)
                if img_md:
                    lines.append(img_md); lines.append("")
                continue

            if looks_jsonish(txt):
                flush_paragraph(); flush_list()
                lines.append(fence(txt, "json")); lines.append("")
                continue

            if is_cli_command_line(txt) or token_ratio(txt) > 0.18:
                flush_paragraph(); flush_list()
                lines.append(fence(txt, guess_code_lang(txt) or "bash")); lines.append("")
                continue

            list_buf and flush_list()
            para_buf.append(txt)

        flush_paragraph()
        flush_list()

    return "\n".join(lines).strip() + "\n"

# ---------- Gemini MD polishing helpers (optional) ----------
def _split_by_page_heading(md_text: str) -> List[str]:
    lines = (md_text or "").splitlines()
    chunks, cur = [], []
    for ln in lines:
        if ln.startswith("## Page ") and cur:
            chunks.append("\n".join(cur).strip() + "\n")
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append("\n".join(cur).strip() + "\n")
    return chunks

def _fix_unclosed_fences(md: str) -> str:
    n = md.count("```")
    return md if n % 2 == 0 else (md.rstrip() + "\n```")

_GEMINI_MD_RULES = """
You are a precise Markdown formatter. Reformat the given Markdown:
- KEEP ALL CONTENT EXACTLY (no additions/removals/translation).
- Preserve headings exactly as is (including lines that start with '## Page ').
- Normalize line wraps and spacing.
- Convert command-like lines to fenced code blocks (bash).
- Convert JSON-like blocks to fenced code blocks (json).
- Separate paragraphs and code blocks with one blank line.
Return Markdown only.
"""

async def gemini_polish_markdown_by_page(md_text: str) -> str:
    global md_model
    if not USE_GEMINI_MD or not md_model or not md_text or len(md_text) < 10:
        return md_text

    chunks = _split_by_page_heading(md_text) or [md_text]
    polished: List[str] = []

    for i, chunk in enumerate(chunks):
        prompt = f"{_GEMINI_MD_RULES}\n\n<md>\n{chunk}\n</md>"
        try:
            resp = await asyncio.to_thread(md_model.generate_content, prompt)
            out = (resp.text or "").strip()
            if not out:
                out = chunk
        except Exception as e:
            logger.warning(f"[Gemini-MD] chunk {i} failed: {e}; keep original.")
            out = chunk
        polished.append(_fix_unclosed_fences(out))

    return ("\n\n".join(polished)).strip() + "\n"

# ---------- Embeddings ----------
async def async_generate_embedding(input_data: List[str]) -> Optional[List[List[float]]]:
    if not input_data:
        return None
    global embedding_http_session
    if embedding_http_session is None:
        embedding_http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY))

    async def _embed_one(text_to_embed: str) -> Optional[List[float]]:
        async def _try_service(url, model, label):
            for attempt in range(3):
                try:
                    timeout = aiohttp.ClientTimeout(total=120)
                    payload = {"model": model, "prompt": text_to_embed}
                    async with embedding_http_session.post(url, json=payload, timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, dict):
                                if isinstance(data.get("embedding"), list):
                                    return data["embedding"]
                                if isinstance(data.get("data"), list) and data["data"]:
                                    first = data["data"][0]
                                    if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                                        return first["embedding"]
                            logger.warning(f"[{label}] unexpected embedding response format")
                        else:
                            logger.warning(f"[{label}] Embedding API status {resp.status}, attempt {attempt+1}")
                            await asyncio.sleep(2 * (attempt + 1))
                except Exception as e:
                    logger.warning(f"[{label}] Connection error to Embedding API: {e}, attempt {attempt+1}")
                    await asyncio.sleep(2 * (attempt + 1))
            return None

        primary_emb = await _try_service(OLLAMA_EMBEDDING_URL, OLLAMA_EMBEDDING_MODEL_NAME, "primary")
        if primary_emb is not None:
            return primary_emb
        if FALLBACK_EMBEDDING_URL and FALLBACK_EMBEDDING_MODEL_NAME:
            logger.warning("Primary embedding failed. Trying fallback.")
            return await _try_service(FALLBACK_EMBEDDING_URL, FALLBACK_EMBEDDING_MODEL_NAME, "fallback")
        return None

    tasks = [_embed_one(text) for text in input_data]
    results = await asyncio.gather(*tasks)
    return [e for e in results if e is not None] or None

async def async_process_text_chunk(chunk_text: str, chunk_metadata: Dict[str, Any]) -> Optional[models.PointStruct]:
    global semaphore_embedding_call
    if not semaphore_embedding_call:
        return None
    async with semaphore_embedding_call:
        chunk_id = chunk_metadata.get('chunk_id')
        if not chunk_id:
            logger.error("Missing chunk_id in metadata")
            return None
        try:
            uuid.UUID(str(chunk_id))
            chunk_id_str = str(chunk_id)
        except Exception:
            fixed_uuid = uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id))
            chunk_id_str = str(fixed_uuid)
            chunk_metadata["chunk_id"] = chunk_id_str
        embedding_results = await async_generate_embedding([chunk_text])
        if not (embedding_results and embedding_results[0]):
            logger.warning(f"Failed to generate embedding for chunk '{chunk_metadata.get('chunk_id')}'")
            return None
        payload = {"pageContent": chunk_text, "metadata": chunk_metadata}
        return models.PointStruct(id=chunk_id_str, vector=embedding_results[0], payload=payload)

# ---------- Job Status & Webhook ----------
async def _read_job_statuses() -> Dict[str, Any]:
    async with job_status_lock:
        if not os.path.exists(JOB_STATUS_FILE):
            return {}
        async with aiofiles.open(JOB_STATUS_FILE, mode='r', encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content) if content else {}

async def _write_job_statuses(statuses: Dict[str, Any]):
    async with job_status_lock:
        tmp_path = JOB_STATUS_FILE + ".tmp"
        async with aiofiles.open(tmp_path, mode='w', encoding="utf-8") as f:
            await f.write(json.dumps(statuses, indent=2, ensure_ascii=False))
        os.replace(tmp_path, JOB_STATUS_FILE)

async def update_job_status(file_path: str, status: str, details: Optional[Dict[str, Any]] = None):
    statuses = await _read_job_statuses()
    now_utc = datetime.now(timezone.utc).isoformat()
    if file_path not in statuses:
        statuses[file_path] = {"created_at": now_utc, "file_path": file_path}
    statuses[file_path]["status"] = status
    statuses[file_path]["updated_at"] = now_utc
    if details:
        statuses[file_path].update(details)
    await _write_job_statuses(statuses)

async def notify_webhook(result_data: ProcessingResponse):
    if not N8N_WEBHOOK_URL:
        logger.warning("[BG] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(N8N_WEBHOOK_URL, json=json.loads(result_data.model_dump_json()))
            if resp.status_code >= 400:
                logger.warning(f"[BG] Webhook ตอบ {resp.status_code}: {resp.text[:200]}")
            else:
                logger.info("[BG] ส่ง Webhook notification สำเร็จ")
    except Exception as e:
        logger.warning(f"[BG] ส่ง Webhook ไม่สำเร็จ: {e}")

# ---------- Library & Qdrant helpers ----------
def parse_page_string(page_str: Optional[str]) -> Set[int]:
    if not page_str:
        return set()
    page_numbers = set()
    parts = page_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end:
                    start, end = end, start
                page_numbers.update(range(start, end + 1))
            else:
                page_numbers.add(int(part))
        except ValueError:
            logger.warning(f"ไม่สามารถแปลงค่าหน้า '{part}' ได้ จะข้ามส่วนนี้ไป")
    return page_numbers

async def get_existing_page_numbers(collection_name: str, source_file_name: str, book_id: Optional[str] = None) -> Set[int]:
    global qdrant_client
    if qdrant_client is None:
        return set()
    existing_pages: Set[int] = set()
    try:
        musts = []
        if book_id:
            musts.append(models.FieldCondition(key="metadata.book_id", match=models.MatchValue(value=book_id)))
        else:
            musts.append(models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source_file_name)))
        source_filter = models.Filter(must=musts)
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(
                qdrant_client.scroll,
                collection_name=collection_name,
                scroll_filter=source_filter,
                limit=250,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            for point in response:
                payload = point.payload or {}
                page_num = (payload.get("metadata") or {}).get("loc", {}).get("pageNumber")
                if isinstance(page_num, int):
                    existing_pages.add(page_num)
            if next_offset is None:
                break
        return existing_pages
    except Exception as e:
        logger.warning(f"เกิดข้อผิดพลาดขณะดึงข้อมูลหน้าที่มีอยู่: {e}. จะถือว่ายังไม่มีหน้าใดๆ")
        return set()

async def fetch_document_info(doc_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    if not LIBRARY_API_TOKEN or not LIBRARY_API_BASE_URL:
        return None
    info_path = f"api/documents/{doc_id}"
    url = urljoin(LIBRARY_API_BASE_URL, info_path)
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"เรียกดูข้อมูลเอกสาร {doc_id} ล้มเหลว: {e}")
        return None

def make_source_label_from_doc_info(doc_info: Dict[str, Any]) -> Optional[str]:
    if not doc_info:
        return None
    title = (doc_info.get("title") or "").strip()
    authors = doc_info.get("authors") or []
    first_author = (authors[0].strip() if authors else "")
    if title and first_author:
        return f"{title} - {first_author}"
    if title:
        return title
    return None

async def search_library_for_book(query: str) -> Optional[List[Dict[str, Any]]]:
    if not LIBRARY_API_BASE_URL:
        return None
    search_url = f"{LIBRARY_API_BASE_URL}/search"
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"} if LIBRARY_API_TOKEN else {}
    params = {"q": query}
    logger.info(f"กำลังค้นหาหนังสือใน Library ด้วยคำว่า: '{query}'...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(search_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and 'files' in data:
                return data['files']
            return []
    except Exception as e:
        logger.error(f"ค้นหา Library ล้มเหลว: {e}")
        return None

async def download_book_from_library(file_id: str) -> Optional[Tuple[str, bytes]]:
    if not LIBRARY_API_BASE_URL:
        return None
    download_path = f"api/documents/{file_id}/download"
    download_url = urljoin(LIBRARY_API_BASE_URL, download_path)
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"} if LIBRARY_API_TOKEN else {}
    logger.info(f"กำลังเริ่มดาวน์โหลดไฟล์ id '{file_id}' จาก URL: {download_url}")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get(download_url, headers=headers)
            resp.raise_for_status()
            content_disp = resp.headers.get('Content-Disposition', '')
            filename = file_id
            if 'filename=' in content_disp:
                filename = content_disp.split('filename=')[-1].strip('"')
            data = resp.content
            logger.info(f"ดาวน์โหลดไฟล์ '{filename}' สำเร็จ")
            return filename, data
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดระหว่างดาวน์โหลดไฟล์ id '{file_id}' จาก Library: {e}")
        return None

async def detect_embedding_dim() -> Optional[int]:
    try:
        emb = await async_generate_embedding(["__probe__"])
        if emb and emb[0]:
            return len(emb[0])
    except Exception as e:
        logger.warning(f"ตรวจจับมิติ embedding ล้มเหลว: {e}")
    return None

async def ensure_qdrant_collection(collection_name: str):
    global qdrant_client
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    try:
        await asyncio.to_thread(qdrant_client.get_collection, collection_name=collection_name)
    except Exception:
        dim = await detect_embedding_dim()
        if not dim:
            dim = QDRANT_VECTOR_SIZE
            logger.warning(f"ใช้ QDRANT_VECTOR_SIZE จาก env แทน (dim={dim})")
        logger.info(f"ไม่พบ Collection '{collection_name}' กำลังสร้าง (dim={dim})...")
        await asyncio.to_thread(
            qdrant_client.create_collection,
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' สร้างสำเร็จแล้ว.")
        try:
            await asyncio.to_thread(
                qdrant_client.create_payload_index,
                collection_name=collection_name,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            await asyncio.to_thread(
                qdrant_client.create_payload_index,
                collection_name=collection_name,
                field_name="metadata.loc.pageNumber",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            await asyncio.to_thread(
                qdrant_client.create_payload_index,
                collection_name=collection_name,
                field_name="pageContent",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WHITESPACE,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )
            await asyncio.to_thread(
                qdrant_client.create_payload_index,
                collection_name=collection_name,
                field_name="metadata.book_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            logger.warning(f"สร้าง payload index ไม่ครบถ้วน: {e}")

async def list_unique_sources_in_collection(collection_name: str) -> List[str]:
    """ดึงรายชื่อค่า metadata.source ที่ไม่ซ้ำภายใน collection"""
    global qdrant_client
    if qdrant_client is None:
        raise RuntimeError("Qdrant client is not initialized")
    unique_sources: Set[str] = set()
    try:
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(
                qdrant_client.scroll,
                collection_name=collection_name,
                limit=250,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in response:
                payload = point.payload or {}
                meta = payload.get("metadata") or {}
                source = meta.get("source")
                if source:
                    unique_sources.add(str(source))
            if next_offset is None:
                break
        return sorted(unique_sources)
    except Exception as e:
        logger.error(f"[Qdrant] list_unique_sources_in_collection error: {e}")
        raise

# ---------- Main processing with Markdown export ----------
async def process_table_element(element, base_metadata: dict) -> Tuple[List[models.PointStruct], int]:
    points_to_upsert: List[models.PointStruct] = []
    attempted = 0
    table_id = str(uuid.uuid4())
    full_table_text = getattr(element, "text", None)
    if full_table_text and full_table_text.strip():
        attempted += 1
        table_metadata = base_metadata.copy()
        table_metadata.update({"table_id": table_id, "content_type": "full_table"})
        table_metadata["chunk_id"] = make_chunk_id(table_metadata, full_table_text)
        point = await async_process_text_chunk(full_table_text, table_metadata)
        if point:
            points_to_upsert.append(point)
    try:
        parsed_rows = parse_markdown_table(full_table_text or "")
        for i, row_data in enumerate(parsed_rows):
            parts = [f"the {k} is '{v}'" for k, v in row_data.items() if str(v).strip()]
            if not parts:
                continue
            attempted += 1
            row_summary_text = f"In this table row, {', '.join(parts)}."
            row_metadata = base_metadata.copy()
            row_metadata.update({"table_id": table_id, "content_type": "table_row", "row_index": i})
            row_metadata["chunk_id"] = make_chunk_id(row_metadata, row_summary_text)
            point = await async_process_text_chunk(row_summary_text, row_metadata)
            if point:
                points_to_upsert.append(point)
    except Exception as e:
        logger.warning(f"Could not parse or serialize markdown table for {base_metadata.get('source')}: {e}")
    return points_to_upsert, attempted

async def process_and_upsert_single_page(
    file_bytes: bytes,
    page_num: int,
    source_label: str,
    collection_name: str,
    book_id: Optional[str] = None,
    image_output_dir: Optional[str] = None,
) -> Tuple[int, int, str]:
    global qdrant_client
    logger.info(f"--- [BG] เริ่มประมวลผลหน้า {page_num} ด้วย Unstructured ---")
    try:
        elements = await asyncio.to_thread(
            partition_pdf,
            file=io.BytesIO(file_bytes),
            strategy="hi_res",
            include_metadata=True,
            infer_table_structure=INFER_TABLE_STRUCTURE,
            extract_images_in_pdf=True,
            image_output_dir_path=image_output_dir,
            languages=OCR_LANGUAGES,
            pages=[page_num],
        )
    except Exception as e:
        logger.error(f"[BG] partition_pdf ล้มเหลวที่หน้า {page_num}: {e}", exc_info=True)
        return 0, 1, f"## Page {page_num}\n\n"

    elements = [el for el in elements if _elem_page_number(el) in (page_num, None)]
    if not elements:
        logger.info(f"--- [BG] Unstructured ไม่พบ elements ใดๆ ในหน้า {page_num} ---")
        return 0, 0, f"## Page {page_num}\n\n"

    page_md_lines: List[str] = [f"## Page {page_num}", ""]
    buf: List[str] = []

    def flush_buf():
        if buf:
            joined = clean_text(" ".join(buf).strip())
            if joined:
                page_md_lines.append(joined)
                page_md_lines.append("")
            buf.clear()

    processing_tasks = []
    attempted_points = 0
    successful_points: List[models.PointStruct] = []

    for i, element in enumerate(elements):
        if is_noise(element):
            continue

        elem_pn = _elem_page_number(element) or page_num
        base_metadata = {"source": source_label, "loc": {"pageNumber": elem_pn, "elementIndex": i}}
        if book_id:
            base_metadata["book_id"] = book_id

        category = getattr(element, "category", None)
        text_val = getattr(element, "text", None)

        if category == "Table":
            flush_buf()
            table_md = render_table(element)
            if table_md:
                page_md_lines.append(table_md)
                page_md_lines.append("")
            processing_tasks.append(process_table_element(element, base_metadata))
            continue

        if category in {"Image", "Figure"}:
            flush_buf()
            img_md = render_image(element, image_output_dir or OCR_MD_OUTPUT_DIR)
            if img_md:
                page_md_lines.append(img_md)
                page_md_lines.append("")
            async def process_image_element(img_element, meta):
                attempted = 1
                try:
                    md = img_element.metadata.to_dict() if getattr(img_element, "metadata", None) else {}
                    img_path = md.get("image_path")
                    if not img_path:
                        paths = md.get("image_paths") or []
                        img_path = paths[0] if paths else None
                    description = "No description available."
                    if img_path and os.path.exists(img_path):
                        with Image.open(img_path) as pil_image:
                            description = await generate_image_caption(pil_image)
                    elif hasattr(img_element, "image_base64") and img_element.image_base64:
                        with Image.open(io.BytesIO(base64.b64decode(img_element.image_base64))) as pil_image:
                            description = await generate_image_caption(pil_image)
                    text_to_embed = f"Image Description: {description}"
                    img_meta = meta.copy()
                    img_meta.update({"content_type": "image_summary"})
                    img_meta["chunk_id"] = make_chunk_id(img_meta, text_to_embed)
                    point = await async_process_text_chunk(text_to_embed, img_meta)
                    return ([point] if point else []), attempted
                except Exception as img_err:
                    logger.error(f"Failed to process image element: {img_err}", exc_info=True)
                    return ([], attempted)
            processing_tasks.append(process_image_element(element, base_metadata))
            continue

        if category == "Title":
            flush_buf()
            title_txt = clean_text(text_val or "")
            if title_txt:
                page_md_lines.append(f"## {title_txt}")
                page_md_lines.append("")
                async def process_text_element(text, meta, content_type: str):
                    attempted = 1
                    text_meta = meta.copy()
                    text_meta.update({"content_type": content_type})
                    text_meta["chunk_id"] = make_chunk_id(text_meta, text)
                    point = await async_process_text_chunk(text, text_meta)
                    return ([point] if point else []), attempted
                processing_tasks.append(process_text_element(title_txt, base_metadata, "Title"))
            continue

        if category == "ListItem":
            flush_buf()
            item_txt = clean_text(text_val or "")
            if item_txt:
                page_md_lines.append(f"- {item_txt}")
                page_md_lines.append("")
                async def process_text_element(text, meta, content_type: str):
                    attempted = 1
                    text_meta = meta.copy()
                    text_meta.update({"content_type": content_type})
                    text_meta["chunk_id"] = make_chunk_id(text_meta, text)
                    point = await async_process_text_chunk(text, text_meta)
                    return ([point] if point else []), attempted
                processing_tasks.append(process_text_element(item_txt, base_metadata, "ListItem"))
            continue

        txt = clean_text(text_val or "")
        if not txt:
            continue
        buf.append(txt)

        async def process_text_element(text, meta, content_type: str):
            attempted = 1
            text_meta = meta.copy()
            text_meta.update({"content_type": content_type})
            text_meta["chunk_id"] = make_chunk_id(text_meta, text)
            point = await async_process_text_chunk(text, text_meta)
            return ([point] if point else []), attempted

        processing_tasks.append(process_text_element(txt, base_metadata, category or "Text"))

    flush_buf()

    results = await asyncio.gather(*processing_tasks, return_exceptions=True)
    for item in results:
        if isinstance(item, Exception):
            logger.error(f"[BG] Element task error (page {page_num}): {item}", exc_info=True)
            continue
    # table: (points, attempted) / text/image: ([point] or None, attempted)
    for item in results:
        if isinstance(item, Exception):
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list):
            pts, att = item
            successful_points.extend([p for p in pts if p])
            attempted_points += att
        else:
            points, attempted = item
            successful_points.extend([p for p in points if p])
            attempted_points += attempted

    failed_count = max(0, attempted_points - len(successful_points))

    if successful_points:
        try:
            await asyncio.to_thread(
                qdrant_client.upsert, collection_name=collection_name, wait=True, points=successful_points
            )
            await mark_activity("qdrant:upsert")
            logger.info(f"[BG] บันทึก {len(successful_points)} points จากหน้า {page_num} ลง Qdrant สำเร็จ")
        except Exception as upsert_err:
            logger.error(f"[BG] Qdrant upsert ล้มเหลวในหน้า {page_num}: {upsert_err}", exc_info=True)

    logger.info(f"--- [BG] จบการประมวลผลหน้า {page_num} ---")
    return len(successful_points), failed_count, ("\n".join(page_md_lines) + "\n")

async def page_worker_with_semaphore(
    file_bytes: bytes,
    page_num: int,
    source_label: str,
    collection_name: str,
    book_id: Optional[str] = None,
    image_output_dir: Optional[str] = None,
) -> Tuple[int, int, str]:
    global page_processing_semaphore
    if page_processing_semaphore is None:
        raise RuntimeError("Page processing semaphore is not initialized.")
    async with page_processing_semaphore:
        return await process_and_upsert_single_page(
            file_bytes, page_num, source_label, collection_name, book_id, image_output_dir
        )

# ---------- Batch mode for large/scanned PDFs ----------
async def _preview_table_pages(file_bytes: bytes, pages: List[int]) -> Set[int]:
    if not PREVIEW_TABLES:
        return set()
    try:
        preview_els = await asyncio.to_thread(
            partition_pdf,
            file=io.BytesIO(file_bytes),
            strategy="fast",
            include_metadata=True,
            pages=pages,
        )
    except Exception as e:
        logger.warning(f"[Batch] Preview fast failed: {e}")
        return set()
    table_pages: Set[int] = set()
    for el in preview_els or []:
        if getattr(el, "category", "") == "Table":
            pn = _elem_page_number(el) or 1
            table_pages.add(pn)
    return table_pages

async def process_batch_pages(
    file_bytes: bytes,
    batch_pages: List[int],
    source_label: str,
    collection_name: str,
    book_id: Optional[str],
    image_output_dir: str,
) -> List[Tuple[int, int, str]]:
    results_per_page: Dict[int, Dict[str, Any]] = {
        p: {"points": [], "attempted": 0, "md_lines": [f"## Page {p}", ""]} for p in batch_pages
    }
    table_pages = await _preview_table_pages(file_bytes, batch_pages) if PREVIEW_TABLES else set()
    non_table_pages = [p for p in batch_pages if p not in table_pages]
    all_elements: List[Any] = []
    try:
        if non_table_pages:
            els = await asyncio.to_thread(
                partition_pdf,
                file=io.BytesIO(file_bytes),
                strategy="hi_res",
                include_metadata=True,
                infer_table_structure=False,
                extract_images_in_pdf=True,
                image_output_dir_path=image_output_dir,
                languages=OCR_LANGUAGES,
                pages=non_table_pages,
            )
            all_elements.extend(els or [])
        if table_pages:
            els = await asyncio.to_thread(
                partition_pdf,
                file=io.BytesIO(file_bytes),
                strategy="hi_res",
                include_metadata=True,
                infer_table_structure=True,
                extract_images_in_pdf=True,
                image_output_dir_path=image_output_dir,
                languages=OCR_LANGUAGES,
                pages=sorted(table_pages),
            )
            all_elements.extend(els or [])
    except Exception as e:
        logger.error(f"[Batch] partition_pdf failed: {e}", exc_info=True)
        return [(0, 0, "\n".join(results_per_page[p]["md_lines"]) + "\n") for p in batch_pages]

    clean_elements = [el for el in all_elements if not (is_noise(el) or is_header_footer(el))]
    clean_elements = sort_by_reading_order(clean_elements)

    grouped: Dict[int, List[Any]] = {}
    for el in clean_elements:
        pn = _elem_page_number(el)
        if pn in results_per_page:
            grouped.setdefault(pn, []).append(el)

    for pn in batch_pages:
        elements = grouped.get(pn, [])
        md_lines = results_per_page[pn]["md_lines"]
        attempted_points = 0

        para_buf: List[str] = []
        list_buf: List[str] = []

        def flush_paragraph():
            nonlocal para_buf
            if para_buf:
                joined = clean_text(" ".join(para_buf).strip())
                if joined:
                    md_lines.append(joined); md_lines.append("")
                para_buf = []

        def flush_list():
            nonlocal list_buf
            if list_buf:
                for li in list_buf:
                    md_lines.append(f"- {clean_text(li)}")
                md_lines.append("")
                list_buf = []

        tasks = []

        for idx, el in enumerate(elements):
            cat = getattr(el, "category", "")
            txt = clean_text(getattr(el, "text", "") or "")

            base_meta = {"source": source_label, "loc": {"pageNumber": pn, "elementIndex": idx}}
            if book_id:
                base_meta["book_id"] = book_id

            if cat == "Title":
                flush_paragraph(); flush_list()
                if is_commandish_title(txt):
                    md_lines.append(fence(txt, guess_code_lang(txt) or "bash")); md_lines.append("")
                    async def _proc():
                        nonlocal attempted_points
                        attempted_points += 1
                        meta = base_meta.copy(); meta.update({"content_type": "Code"})
                        meta["chunk_id"] = make_chunk_id(meta, txt)
                        return await async_process_text_chunk(txt, meta)
                    tasks.append(_proc())
                else:
                    md_lines.append(f"## {txt}"); md_lines.append("")
                    async def _proc():
                        nonlocal attempted_points
                        attempted_points += 1
                        meta = base_meta.copy(); meta.update({"content_type": "Title"})
                        meta["chunk_id"] = make_chunk_id(meta, txt)
                        return await async_process_text_chunk(txt, meta)
                    tasks.append(_proc())
                continue

            if cat == "ListItem":
                flush_paragraph()
                if txt:
                    list_buf.append(txt)
                    async def _proc():
                        nonlocal attempted_points
                        attempted_points += 1
                        meta = base_meta.copy(); meta.update({"content_type": "ListItem"})
                        meta["chunk_id"] = make_chunk_id(meta, txt)
                        return await async_process_text_chunk(txt, meta)
                    tasks.append(_proc())
                continue

            if cat == "Table":
                flush_paragraph(); flush_list()
                tbl = render_table(el)
                if tbl:
                    md_lines.append(tbl); md_lines.append("")
                tasks.append(process_table_element(el, base_meta))
                continue

            if cat in {"Image", "Figure"}:
                flush_paragraph(); flush_list()
                img_md = render_image(el, image_output_dir)
                if img_md:
                    md_lines.append(img_md); md_lines.append("")
                async def _proc():
                    nonlocal attempted_points
                    attempted_points += 1
                    description = "No description available."
                    try:
                        md = el.metadata.to_dict() if getattr(el, "metadata", None) else {}
                        img_path = md.get("image_path") or (md.get("image_paths") or [None])[0]
                        if img_path and os.path.exists(img_path):
                            with Image.open(img_path) as pil:
                                description = await generate_image_caption(pil)
                        elif hasattr(el, "image_base64") and el.image_base64:
                            with Image.open(io.BytesIO(base64.b64decode(el.image_base64))) as pil:
                                description = await generate_image_caption(pil)
                    except Exception as e:
                        logger.warning(f"[Batch] image caption failed (page {pn}): {e}")
                    text_to_embed = f"Image Description: {description}"
                    meta = base_meta.copy(); meta.update({"content_type": "image_summary"})
                    meta["chunk_id"] = make_chunk_id(meta, text_to_embed)
                    return await async_process_text_chunk(text_to_embed, meta)
                tasks.append(_proc())
                continue

            if txt:
                if looks_jsonish(txt) or is_cli_command_line(txt) or token_ratio(txt) > 0.18:
                    flush_paragraph(); flush_list()
                    lang = "json" if looks_jsonish(txt) else (guess_code_lang(txt) or "bash")
                    md_lines.append(fence(txt, lang)); md_lines.append("")
                    async def _proc():
                        nonlocal attempted_points
                        attempted_points += 1
                        meta = base_meta.copy(); meta.update({"content_type": "Code"})
                        meta["chunk_id"] = make_chunk_id(meta, txt)
                        return await async_process_text_chunk(txt, meta)
                    tasks.append(_proc())
                else:
                    list_buf and flush_list()
                    para_buf.append(txt)
                    async def _proc():
                        nonlocal attempted_points
                        attempted_points += 1
                        meta = base_meta.copy(); meta.update({"content_type": cat or "Text"})
                        meta["chunk_id"] = make_chunk_id(meta, txt)
                        return await async_process_text_chunk(txt, meta)
                    tasks.append(_proc())

        flush_paragraph()
        flush_list()

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        points: List[models.PointStruct] = []
        for item in gathered:
            if isinstance(item, Exception):
                logger.warning(f"[Batch] task error page {pn}: {item}")
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list):
                pts, att = item
                points.extend([p for p in pts if p])
                attempted_points += att
            elif item:
                points.append(item)

        results_per_page[pn]["points"] = points
        results_per_page[pn]["attempted"] = attempted_points

    all_points: List[models.PointStruct] = []
    for p in batch_pages:
        all_points.extend(results_per_page[p]["points"])
    if all_points:
        try:
            await asyncio.to_thread(
                qdrant_client.upsert, collection_name=collection_name, wait=True, points=all_points
            )
            await mark_activity("qdrant:upsert:batch")
            logger.info(f"[Batch] Upsert {len(all_points)} points for pages {batch_pages[0]}..{batch_pages[-1]} OK")
        except Exception as e:
            logger.error(f"[Batch] Qdrant upsert failed: {e}", exc_info=True)

    results: List[Tuple[int, int, str]] = []
    for p in batch_pages:
        attempted = results_per_page[p]["attempted"]
        success = len(results_per_page[p]["points"])
        failed = max(0, attempted - success)
        page_md = "\n".join(results_per_page[p]["md_lines"]) + "\n"
        results.append((success, failed, page_md))
    return results

# ---------- Background processing ----------
async def process_pdf_in_background(
    file_path_key: str, file_bytes: bytes, file_name: str, pages_str: Optional[str] = None, custom_collection_name: Optional[str] = None
):
    original_file_name = file_name.strip()
    source_label = os.path.splitext(original_file_name)[0]
    book_id: Optional[str] = None
    try:
        doc_info = await fetch_document_info(file_path_key)
        label = make_source_label_from_doc_info(doc_info) if doc_info else None
        if label:
            source_label = label
        if doc_info and "id" in doc_info:
            book_id = str(doc_info["id"])
        elif str(file_path_key).isdigit():
            book_id = str(file_path_key)
    except Exception as e:
        logger.warning(f"Could not fetch doc info for {file_path_key}: {e}")

    collection_name = custom_collection_name.strip().lower() if custom_collection_name else "".join(
        c for c in source_label if c.isalnum() or c in ['-', '_']
    ).lower() or f"doc_{uuid.uuid4().hex}"

    await update_job_status(file_path_key, "processing", {"file_path": original_file_name, "collection_name": collection_name})

    total_successful_chunks = 0
    total_failed_chunks = 0
    final_response = None
    md_output_path: Optional[str] = None

    try:
        await ensure_qdrant_collection(collection_name)

        doc_title = _safe_filename(source_label or os.path.splitext(original_file_name)[0])
        image_output_dir = os.path.join(OCR_MD_OUTPUT_DIR, f"{doc_title}_images")
        os.makedirs(image_output_dir, exist_ok=True)

        total_pages_in_doc = 0

        if _fitz_available:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    total_pages_in_doc = len(doc)
            except Exception as e:
                logger.warning(f"Could not use PyMuPDF for page count: {e}. Falling back to other methods.")

        if total_pages_in_doc == 0 and _pypdf_available:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                total_pages_in_doc = len(reader.pages)
            except Exception as e:
                logger.warning(f"Could not use pypdf for page count: {e}")

        if total_pages_in_doc == 0:
            try:
                elements = await asyncio.to_thread(partition_pdf, file=io.BytesIO(file_bytes), strategy="fast")
                if elements:
                    page_numbers = [getattr(el.metadata, "page_number", None) or getattr(el, "metadata", {}).get("page_number") for el in elements]
                    page_numbers = [p for p in page_numbers if isinstance(p, int)]
                    total_pages_in_doc = max(page_numbers) if page_numbers else 0
                if total_pages_in_doc == 0:
                    raise RuntimeError("Page count detection failed (no elements with page_number).")
            except Exception as e:
                await update_job_status(file_path_key, "failed", {"error": f"Could not determine page count: {e}"})
                return

        requested_pages = parse_page_string(pages_str)
        existing_pages = await get_existing_page_numbers(collection_name, source_label, book_id)
        all_doc_pages = set(range(1, total_pages_in_doc + 1))
        pages_to_process = (requested_pages - existing_pages) if requested_pages else (all_doc_pages - existing_pages)

        if not pages_to_process:
            final_response = ProcessingResponse(
                collection_name=collection_name,
                status="skipped",
                processed_chunks=0,
                failed_chunks=0,
                message="All requested pages already processed.",
                file_name=original_file_name
            )
        else:
            pages_to_iterate = sorted([p for p in pages_to_process if 1 <= p <= total_pages_in_doc])
            logger.info(f"[BG] จะทำการประมวลผลหน้าใหม่ {len(pages_to_iterate)} หน้า (สูงสุด {PAGE_CONCURRENCY} หน้าพร้อมกัน): {pages_to_iterate}")

            results: List[Tuple[int, int, str]] = []
            if LARGE_PDF_MODE:
                for batch in _chunk_list(pages_to_iterate, BATCH_PAGE_SIZE):
                    logger.info(f"[BG] Batch processing pages: {batch[0]}..{batch[-1]} (size={len(batch)})")
                    batch_results = await process_batch_pages(
                        file_bytes=file_bytes,
                        batch_pages=batch,
                        source_label=source_label,
                        collection_name=collection_name,
                        book_id=book_id,
                        image_output_dir=image_output_dir,
                    )
                    results.extend(batch_results)
            else:
                tasks = [
                    page_worker_with_semaphore(file_bytes, page_num, source_label, collection_name, book_id, image_output_dir)
                    for page_num in pages_to_iterate
                ]
                results = await asyncio.gather(*tasks)

            total_successful_chunks = sum(res[0] for res in results)
            total_failed_chunks = sum(res[1] for res in results)

            if SAVE_MD_ENABLED:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                md_filename = f"{doc_title}_{ts}.md"
                md_output_path = os.path.join(OCR_MD_OUTPUT_DIR, md_filename)

                md_text = ""
                if MD_REBUILD_WHOLEDOC:
                    md_text = await rebuild_markdown_whole_doc(
                        file_bytes=file_bytes,
                        pages_to_iterate=pages_to_iterate,
                        source_label=source_label,
                        image_output_dir=image_output_dir,
                    )
                if not md_text:
                    combined_md = [f"# {source_label or original_file_name}", ""]
                    for (_, _, page_md) in results:
                        combined_md.append(page_md)
                    md_text = "\n".join(combined_md)

                # ให้ Gemini ช่วย polish (ถ้าเปิด USE_GEMINI_MD)
                try:
                    md_text = await gemini_polish_markdown_by_page(md_text)
                except Exception as e:
                    logger.warning(f"[Gemini-MD] polishing failed, keep original: {e}")

                try:
                    async with aiofiles.open(md_output_path, "w", encoding="utf-8") as f:
                        await f.write(md_text)
                    logger.info(f"[BG] บันทึก Markdown: {md_output_path}")
                except Exception as e:
                    logger.warning(f"[BG] บันทึก Markdown ล้มเหลว: {e}")
                    md_output_path = None

            status_code = "success" if total_successful_chunks > 0 else "warning"
            extra = f" Markdown saved to: {md_output_path}" if md_output_path else ""
            message = f"Processed {total_successful_chunks} new chunks into collection '{collection_name}'.{extra}"
            final_response = ProcessingResponse(
                collection_name=collection_name,
                status=status_code,
                processed_chunks=total_successful_chunks,
                failed_chunks=total_failed_chunks,
                message=message,
                file_name=original_file_name
            )

        await update_job_status(file_path_key, "completed", {"result": json.loads(final_response.model_dump_json())})
    except Exception as e:
        logger.error(f"[BG Task] Critical error: {e}", exc_info=True)
        error_message = f"Critical processing error: {e}"
        final_response = ProcessingResponse(
            collection_name=collection_name,
            status="failed",
            processed_chunks=0,
            failed_chunks=0,
            message=error_message,
            file_name=original_file_name
        )
        await update_job_status(file_path_key, "failed", {"error": error_message})
    finally:
        if final_response:
            await notify_webhook(final_response)

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_model, md_model, qdrant_client, semaphore_embedding_call, page_processing_semaphore, embedding_http_session, idle_watchdog_task
    logger.info("Application startup initiated...")

    required_vars = [GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, LIBRARY_API_TOKEN]
    if not all(required_vars):
        logger.critical("Missing required environment variables. Shutting down.")
        sys.exit(1)

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        if USE_GEMINI_MD:
            md_model = genai.GenerativeModel(
                GEMINI_MD_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_MD_TEMPERATURE,
                    max_output_tokens=GEMINI_MD_MAX_OUTPUT_TOKENS,
                ),
            )
        logger.info("Gemini Models connected. (vision: on, md: %s)", "on" if USE_GEMINI_MD else "off")
    except Exception as e:
        logger.critical(f"Failed to connect to Gemini: {e}", exc_info=True)
        sys.exit(1)

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=60, port=443, https=True)
        await asyncio.to_thread(qdrant_client.get_collections)
        logger.info("Qdrant Cloud Client connected.")
    except Exception as e:
        logger.critical(f"Failed to connect to Qdrant: {e}", exc_info=True)
        sys.exit(1)

    semaphore_embedding_call = asyncio.Semaphore(CONCURRENCY)
    page_processing_semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    embedding_http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY))
    logger.info("Service startup complete.")

    await mark_activity("startup")
    asyncio.create_task(notify_startup())
    idle_watchdog_task = asyncio.create_task(idle_watchdog())

    yield

    if idle_watchdog_task:
        idle_watchdog_task.cancel()
        await asyncio.gather(idle_watchdog_task, return_exceptions=True)

    if embedding_http_session:
        await asyncio.wait_for(embedding_http_session.close(), timeout=10)

    logger.info("Application shutdown.")

# ---------- FastAPI app ----------
app = FastAPI(
    title="บริการประมวลผล PDF (Unstructured + Multi-modal RAG)",
    description="ใช้ Unstructured framework เพื่อวิเคราะห์โครงสร้างเอกสารและสร้าง Index แบบ Multi-modal พร้อมบันทึกผลเป็น Markdown",
    version="4.5.0",
    lifespan=lifespan
)

# ---------- Endpoints ----------
@app.post("/process_pdf/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_pdf_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...), pages: Optional[str] = Form(None), collection_name: Optional[str] = Form(None)
):
    await mark_activity("endpoint:process_pdf")
    file_path_key = file.filename
    file_bytes = await file.read()
    await update_job_status(file_path_key, "queued", {"file_path": file_path_key})
    background_tasks.add_task(
        process_pdf_in_background, file_path_key, file_bytes, file.filename, pages, collection_name
    )
    return AcknowledgementResponse(message="Task accepted.", file_path=file_path_key, task_status="queued")

@app.post("/process_from_library/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_from_library(request: LibrarySearchRequest, background_tasks: BackgroundTasks):
    await mark_activity("endpoint:process_from_library")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ query")
    found_files = await search_library_for_book(request.query)
    if not found_files:
        raise HTTPException(status_code=404, detail=f"ไม่พบหนังสือใน Library: '{request.query}'")
    file_to_process = found_files[0]
    file_id = file_to_process.get('id') or file_to_process.get('path')
    if not file_id:
        raise HTTPException(status_code=500, detail="ข้อมูลจาก Library API ไม่มี 'id' หรือ 'path' ของไฟล์")
    download_result = await download_book_from_library(str(file_id))
    if download_result is None:
        raise HTTPException(status_code=500, detail=f"ดาวน์โหลดไฟล์ id '{file_id}' จาก Library ล้มเหลว")
    file_name, file_bytes = download_result
    await update_job_status(str(file_id), "queued", {"file_path": file_name})
    background_tasks.add_task(
        process_pdf_in_background, str(file_id), file_bytes, file_name, request.pages, request.collection_name
    )
    return AcknowledgementResponse(message="Task accepted.", file_path=str(file_id), task_status="queued")

@app.post("/process_by_id/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_by_file_path(request: ProcessByPathRequest, background_tasks: BackgroundTasks):
    await mark_activity("endpoint:process_by_id")
    file_id = request.file_path
    if not file_id or not str(file_id).strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ file_path")
    download_result = await download_book_from_library(str(file_id))
    if download_result is None:
        raise HTTPException(status_code=500, detail=f"ดาวน์โหลดไฟล์ id '{file_id}' จาก Library ล้มเหลว")
    file_name, file_bytes = download_result
    await update_job_status(str(file_id), "queued", {"file_path": file_name})
    background_tasks.add_task(
        process_pdf_in_background, str(file_id), file_bytes, file_name, request.pages, request.collection_name
    )
    return AcknowledgementResponse(message="Task accepted.", file_path=str(file_id), task_status="queued")

@app.get("/status", response_model=JobStatusResponse)
async def get_job_status(file_path: str = Query(..., description="File Path from submission")):
    await mark_activity("endpoint:get_status")
    statuses = await _read_job_statuses()
    job_details = statuses.get(file_path)
    if not job_details:
        raise HTTPException(status_code=404, detail=f"Status for '{file_path}' not found.")
    return JobStatusResponse(file_path=file_path, details=job_details)

@app.get("/collections/{collection_name}/sources", response_model=SourceListResponse)
async def get_sources_in_collection(
    collection_name: str = Path(..., description="ชื่อของ Collection ที่ต้องการตรวจสอบ")
):
    await mark_activity("endpoint:get_sources_in_collection")
    try:
        sources = await list_unique_sources_in_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error listing sources: {e}")
    return SourceListResponse(collection_name=collection_name, source_count=len(sources), sources=sources)

# --- Cleanup by book_id ---
async def _qdrant_build_book_filter(book_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="metadata.book_id", match=models.MatchValue(value=book_id))])

async def _qdrant_count_by_filter(collection_name: str, qfilter: models.Filter) -> int:
    global qdrant_client
    def _count():
        try:
            res = qdrant_client.count(collection_name=collection_name, count_filter=qfilter, exact=True)
        except TypeError:
            try:
                res = qdrant_client.count(collection_name=collection_name, query_filter=qfilter, exact=True)
            except TypeError:
                res = qdrant_client.count(collection_name=collection_name, filter=qfilter, exact=True)
        return int(getattr(res, "count", 0))
    return await asyncio.to_thread(_count)

async def _qdrant_delete_by_filter(collection_name: str, qfilter: models.Filter) -> None:
    global qdrant_client
    def _delete():
        qdrant_client.delete(collection_name=collection_name, points_selector=qfilter, wait=True)
    return await asyncio.to_thread(_delete)

async def _list_all_collections() -> List[str]:
    global qdrant_client
    def _list_names():
        cols = qdrant_client.get_collections()
        return [c.name for c in getattr(cols, "collections", [])]
    return await asyncio.to_thread(_list_names)

@app.delete("/cleanup/by_book_id", response_model=CleanupByBookIdResponse)
async def cleanup_by_book_id(request: CleanupByBookIdRequest):
    global qdrant_client
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client is not available.")
    await mark_activity("endpoint:cleanup_by_book_id")

    if not request.book_id or not str(request.book_id).strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ book_id")

    if request.collection_name:
        collections = [request.collection_name.strip()]
    else:
        try:
            collections = await _list_all_collections()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ไม่สามารถดึงรายชื่อ collections: {e}")

    if not collections:
        return CleanupByBookIdResponse(total_matched=0, total_deleted=0, dry_run=request.dry_run, details=[])

    qfilter = await _qdrant_build_book_filter(request.book_id)

    total_matched = 0
    total_deleted = 0
    details: List[Dict[str, Any]] = []

    for col in collections:
        try:
            matched = await _qdrant_count_by_filter(col, qfilter)
        except Exception as e:
            details.append({"collection": col, "matched": 0, "deleted": 0, "status": "error", "error": str(e)})
            continue

        deleted = 0
        if not request.dry_run and matched > 0:
            try:
                await _qdrant_delete_by_filter(col, qfilter)
                deleted = matched
                await mark_activity(f"cleanup:deleted:{col}")
            except Exception as e:
                details.append({"collection": col, "matched": matched, "deleted": 0, "status": "error", "error": str(e)})
                total_matched += matched
                continue

        total_matched += matched
        total_deleted += deleted
        details.append({
            "collection": col,
            "matched": matched,
            "deleted": deleted if not request.dry_run else 0,
            "status": "ok" if (request.dry_run or deleted == matched) else "partial"
        })

    return CleanupByBookIdResponse(
        total_matched=total_matched,
        total_deleted=total_deleted if not request.dry_run else 0,
        dry_run=request.dry_run,
        details=details
    )

@app.get("/")
async def root():
    await mark_activity("endpoint:root")
    return {"message": "บริการประมวลผล PDF (Unstructured + Multi-modal) กำลังทำงาน. ใช้ /docs สำหรับเอกสาร API."}

@app.get("/health")
async def health_check():
    global qdrant_client
    await mark_activity("endpoint:health")
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }
    try:
        if qdrant_client:
            await asyncio.to_thread(qdrant_client.get_collections)
            health_status["components"]["qdrant"] = {"status": "healthy"}
        else:
            health_status["components"]["qdrant"] = {"status": "unhealthy", "reason": "Qdrant client not initialized"}
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["qdrant"] = {"status": "unhealthy", "reason": str(e)}
        health_status["status"] = "unhealthy"

    missing_vars = []
    required_vars = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "LIBRARY_API_TOKEN": LIBRARY_API_TOKEN
    }
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)

    if missing_vars:
        health_status["components"]["environment"] = {"status": "unhealthy", "missing_variables": missing_vars}
        health_status["status"] = "unhealthy"
    else:
        health_status["components"]["environment"] = {"status": "healthy"}

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/config")
async def get_current_config():
    await mark_activity("endpoint:get_config")
    return {
        "workers": {"page_concurrency": PAGE_CONCURRENCY, "embedding_concurrency": CONCURRENCY},
        "storage": {"qdrant_vector_size": QDRANT_VECTOR_SIZE},
        "apis": {
            "ollama_embedding_url": OLLAMA_EMBEDDING_URL,
            "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL_NAME,
            "fallback_embedding_url": FALLBACK_EMBEDDING_URL,
            "fallback_embedding_model": FALLBACK_EMBEDDING_MODEL_NAME,
            "library_api_base_url": LIBRARY_API_BASE_URL
        },
        "md_export": {
            "enabled": SAVE_MD_ENABLED,
            "output_dir": OCR_MD_OUTPUT_DIR,
            "rebuild_whole_doc": MD_REBUILD_WHOLEDOC,
            "cli_hints": MD_CLI_HINTS
        },
        "large_pdf_mode": {
            "enabled": LARGE_PDF_MODE,
            "batch_page_size": BATCH_PAGE_SIZE,
            "preview_tables": PREVIEW_TABLES
        },
        "gemini": {
            "use_caption": USE_GEMINI_CAPTION,
            "use_md": USE_GEMINI_MD,
            "md_model": GEMINI_MD_MODEL
        }
    }

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8999)