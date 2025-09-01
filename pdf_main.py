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
import tempfile

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

import torch  # เพิ่มเพื่อเช็ค/ตั้งค่า device/dtype

import google.generativeai as genai
from qdrant_client import QdrantClient, models
from PIL import Image
import httpx
import aiofiles
from urllib.parse import urljoin
import aiohttp

# Marker (แทน Unstructured)
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

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

# Qdrant gRPC options
QDRANT_USE_GRPC = os.getenv("QDRANT_USE_GRPC", "false").lower() in ("1", "true", "yes")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

# Marker options (config ได้ผ่าน ENV)
MARKER_USE_LLM = os.getenv("MARKER_USE_LLM", "true").lower() in ("1", "true", "yes")
MARKER_FORCE_OCR = os.getenv("MARKER_FORCE_OCR", "false").lower() in ("1", "true", "yes")
MARKER_REDO_INLINE_MATH = os.getenv("MARKER_REDO_INLINE_MATH", "false").lower() in ("1", "true", "yes")
MARKER_DISABLE_IMAGE_EXTRACTION = os.getenv("MARKER_DISABLE_IMAGE_EXTRACTION", "false").lower() in ("1", "true", "yes")

# New: Markdown export
OCR_MD_OUTPUT_DIR = os.getenv("OCR_MD_OUTPUT_DIR", "ocr_md_outputs")
SAVE_MD_ENABLED = os.getenv("SAVE_MD_ENABLED", "true").lower() in ("1", "true", "yes")
os.makedirs(OCR_MD_OUTPUT_DIR, exist_ok=True)

# New: Image export
IMAGE_OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "ocr_images")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

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
qdrant_client: Optional[QdrantClient] = None
semaphore_embedding_call: Optional[asyncio.Semaphore] = None
page_processing_semaphore: Optional[asyncio.Semaphore] = None
embedding_http_session: Optional[aiohttp.ClientSession] = None

# --- Marker artifacts/load once ---
MARKER_ARTIFACTS: Optional[dict] = None
MARKER_CONVERT_LOCK = asyncio.Lock()  # กัน race ในบางสภาพแวดล้อม

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

# ---------- Helpers for filename ----------
def _safe_filename(name: str) -> str:
    keep = "-_.() []"
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in keep).strip()
    return cleaned or f"doc_{uuid.uuid4().hex}"

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

# ---------- Marker helpers ----------
def _marker_build_config(output_format: str, page_range: Optional[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "output_format": output_format,
    }
    if page_range:
        cfg["page_range"] = page_range
    if MARKER_USE_LLM:
        cfg["use_llm"] = True
    if MARKER_FORCE_OCR:
        cfg["force_ocr"] = True
    if MARKER_REDO_INLINE_MATH:
        cfg["redo_inline_math"] = True
    if MARKER_DISABLE_IMAGE_EXTRACTION:
        cfg["disable_image_extraction"] = True
    return cfg

def _marker_convert_sync(file_bytes: bytes, output_format: str, page_range: Optional[str]):
    # เขียนไฟล์ชั่วคราวเพื่อให้ Marker อ่าน
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        cfg = _marker_build_config(output_format, page_range)
        config_parser = ConfigParser(cfg)
        converter = PdfConverter(
            artifact_dict=MARKER_ARTIFACTS,  # ใช้โมเดลที่โหลดครั้งเดียว
            config=config_parser.generate_config_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),  # ใช้ GOOGLE_API_KEY
        )
        rendered = converter(tmp_path)
        text, metadata, images = text_from_rendered(rendered)
        return text, metadata, images
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

async def marker_convert(file_bytes: bytes, output_format: str, page_range: Optional[str]):
    async with MARKER_CONVERT_LOCK:  # กัน race (ถ้าต้องการ throughput สูง ค่อยพิจารณาปลด)
        return await asyncio.to_thread(_marker_convert_sync, file_bytes, output_format, page_range)

def _walk_images_from_blocks(block: Dict[str, Any], collector: List[Tuple[str, str]]):
    imgs = block.get("images") or {}
    for blk_id, b64 in imgs.items():
        collector.append((blk_id, b64))
    for ch in (block.get("children") or []):
        _walk_images_from_blocks(ch, collector)

def extract_images_from_json_blocks(blocks: List[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    """
    คืน [(block_id, base64, page_index)]
    """
    results: List[Tuple[str, str, int]] = []
    for page_idx, page_block in enumerate(blocks):
        tmp: List[Tuple[str, str]] = []
        _walk_images_from_blocks(page_block, tmp)
        for blk_id, b64 in tmp:
            results.append((blk_id, b64, page_idx))
    return results

# ---------- Image caption ----------
async def generate_image_caption(image: Image.Image) -> str:
    """ใช้ Gemini Vision เพื่อสร้างคำอธิบายรูปภาพ"""
    global vision_model
    if not vision_model:
        logger.warning("Vision model is not initialized. Cannot generate image caption.")
        return "No description available."
    try:
        prompt = [IMAGE_CAPTION_PROMPT, image]
        response = await asyncio.to_thread(vision_model.generate_content, prompt)
        return response.text or "Image could not be described."
    except Exception as e:
        logger.error(f"Error generating image caption: {e}")
        return "Error in image description generation."

# ---------- Utils ----------
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

# ---------- Embeddings ----------
async def async_generate_embedding(input_data: List[str]) -> Optional[List[List[float]]]:
    """เรียก Embedding API แบบ async ด้วย aiohttp พร้อม Fallback"""
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
    """สร้าง Vector Embedding และจัดรูปแบบเป็น PointStruct"""
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
    info_path = f"api/books/{doc_id}"
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
    """ดาวน์โหลดไฟล์จาก Library API แบบใช้ id (/api/books/{id}/download)"""
    if not LIBRARY_API_BASE_URL:
        return None
    download_path = f"api/books/{file_id}/download"
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
            # fallback มิติจาก env หากตรวจจับไม่ได้
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
    global qdrant_client
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client is not available.")
    unique_sources = set()
    try:
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(
                qdrant_client.scroll,
                collection_name=collection_name,
                limit=250,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            for point in response:
                payload = point.payload or {}
                source = (payload.get("metadata") or {}).get("source")
                if source:
                    unique_sources.add(source)
            if next_offset is None:
                break
        return sorted(list(unique_sources))
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดขณะ list sources จาก collection '{collection_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or an error occurred.")

# ---------- Main processing with Marker ----------
async def process_and_upsert_single_page(
    file_bytes: bytes, page_num: int, source_label: str, collection_name: str, book_id: Optional[str] = None
) -> Tuple[int, int, str]:
    """
    ประมวลผลหน้าเดียวด้วย Marker:
    - ดึง chunks ของหน้า → สร้าง embedding → upsert
    - ดึงรูปจาก JSON → เซฟไฟล์ → Gemini Vision caption → upsert
    - สร้าง Markdown ของหน้านั้นเพื่อรวมไฟล์ MD ภายหลัง
    """
    global qdrant_client
    attempted_points = 0
    points_collected: List[models.PointStruct] = []

    # 1) CHUNKS (สำหรับ RAG)
    try:
        chunks, _, _ = await marker_convert(file_bytes, output_format="chunks", page_range=str(page_num))
        chunk_tasks = []
        for i, ch in enumerate(chunks or []):
            content = (ch.get("html") or ch.get("text") or "").strip()
            if not content:
                continue
            attempted_points += 1
            meta: Dict[str, Any] = (ch.get("metadata") or {}).copy()
            meta["source"] = source_label
            meta["content_type"] = meta.get("block_type") or "chunk"
            meta["loc"] = {"pageNumber": page_num, "elementIndex": i}
            if book_id:
                meta["book_id"] = book_id
            meta["chunk_id"] = make_chunk_id(meta, content)

            async def _proc(text: str, m: Dict[str, Any]):
                return await async_process_text_chunk(text, m)

            chunk_tasks.append(_proc(content, meta))

        if chunk_tasks:
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            for res in chunk_results:
                if isinstance(res, models.PointStruct):
                    points_collected.append(res)
                elif isinstance(res, Exception):
                    logger.error(f"[Marker] chunk task error (page {page_num}): {res}", exc_info=True)
    except Exception as e:
        logger.error(f"[Marker] Convert chunks fail (page {page_num}): {e}", exc_info=True)

    # 2) IMAGES (JSON → save → caption → upsert)
    try:
        blocks, _, _ = await marker_convert(file_bytes, output_format="json", page_range=str(page_num))
        images_info = extract_images_from_json_blocks(blocks or [])
        image_tasks = []
        save_root = os.path.join(IMAGE_OUTPUT_DIR, _safe_filename(source_label), f"page_{page_num}")
        os.makedirs(save_root, exist_ok=True)

        for idx, (blk_id, b64, page_idx) in enumerate(images_info):
            attempted_points += 1

            async def _proc_image(_b64: str, _idx: int):
                try:
                    raw = base64.b64decode(_b64)
                    with Image.open(io.BytesIO(raw)) as pil:
                        fname = f"img_{_idx:03d}.png"
                        fpath = os.path.join(save_root, fname)
                        pil.convert("RGB").save(fpath, format="PNG")

                        caption = await generate_image_caption(pil)

                    text_to_embed = f"Image Description: {caption}"
                    meta: Dict[str, Any] = {
                        "source": source_label,
                        "content_type": "image_caption",
                        "image_path": fpath,
                        "loc": {"pageNumber": page_num, "elementIndex": _idx},
                    }
                    if book_id:
                        meta["book_id"] = book_id
                    meta["chunk_id"] = make_chunk_id(meta, text_to_embed)
                    return await async_process_text_chunk(text_to_embed, meta)
                except Exception as ex:
                    logger.error(f"[Marker] image processing failed (page {page_num}, idx {_idx}): {ex}", exc_info=True)
                    return None

            image_tasks.append(_proc_image(b64, idx))

        if image_tasks:
            image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
            for res in image_results:
                if isinstance(res, models.PointStruct):
                    points_collected.append(res)
                elif isinstance(res, Exception):
                    logger.error(f"[Marker] image task error (page {page_num}): {res}", exc_info=True)
    except Exception as e:
        logger.error(f"[Marker] Convert JSON(images) fail (page {page_num}): {e}", exc_info=True)

    # 3) UPSERT (รวมทั้ง chunks+images)
    failed_count = max(0, attempted_points - len(points_collected))
    if points_collected:
        try:
            await asyncio.to_thread(
                qdrant_client.upsert,
                collection_name=collection_name,
                wait=True,
                points=points_collected
            )
            await mark_activity("qdrant:upsert")
            logger.info(f"[BG] บันทึก {len(points_collected)} points จากหน้า {page_num} ลง Qdrant สำเร็จ")
        except Exception as upsert_err:
            logger.error(f"[BG] Qdrant upsert ล้มเหลวในหน้า {page_num}: {upsert_err}", exc_info=True)

    # 4) MARKDOWN (เฉพาะเพื่อรวมไฟล์ .md ส่งออก)
    page_md = f"## Page {page_num}\n\n"
    try:
        md_text, _, _ = await marker_convert(file_bytes, output_format="markdown", page_range=str(page_num))
        if isinstance(md_text, str) and md_text.strip():
            page_md += md_text.strip() + "\n"
    except Exception as e:
        logger.warning(f"[Marker] Convert markdown fail (page {page_num}): {e}")

    return len(points_collected), failed_count, page_md

async def page_worker_with_semaphore(
    file_bytes: bytes, page_num: int, source_label: str, collection_name: str, book_id: Optional[str] = None
) -> Tuple[int, int, str]:
    global page_processing_semaphore
    if page_processing_semaphore is None:
        raise RuntimeError("Page processing semaphore is not initialized.")
    async with page_processing_semaphore:
        return await process_and_upsert_single_page(file_bytes, page_num, source_label, collection_name, book_id)

async def _count_pages_with_marker_fallback(file_bytes: bytes) -> int:
    """
    ใช้ Marker (json) เพื่อหา page count (fallback สุดท้าย ถ้าไม่มี fitz/pypdf)
    """
    try:
        blocks, _, _ = await marker_convert(file_bytes, output_format="json", page_range=None)
        if isinstance(blocks, list):
            return len(blocks)
    except Exception as e:
        logger.warning(f"Marker fallback page count failed: {e}")
    return 0

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
            total_pages_in_doc = await _count_pages_with_marker_fallback(file_bytes)
            if total_pages_in_doc == 0:
                await update_job_status(file_path_key, "failed", {"error": "Could not determine page count"})
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

            tasks = [page_worker_with_semaphore(file_bytes, page_num, source_label, collection_name, book_id) for page_num in pages_to_iterate]
            results = await asyncio.gather(*tasks)

            total_successful_chunks = sum(res[0] for res in results)
            total_failed_chunks = sum(res[1] for res in results)

            # รวม Markdown และบันทึกไฟล์
            if SAVE_MD_ENABLED:
                doc_title = _safe_filename(source_label or os.path.splitext(original_file_name)[0])
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                md_filename = f"{doc_title}_{ts}.md"
                md_output_path = os.path.join(OCR_MD_OUTPUT_DIR, md_filename)
                combined_md = [f"# {source_label or original_file_name}", ""]
                for (_, _, page_md) in results:
                    combined_md.append(page_md)
                try:
                    async with aiofiles.open(md_output_path, "w", encoding="utf-8") as f:
                        await f.write("\n".join(combined_md))
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
    global vision_model, qdrant_client, semaphore_embedding_call, page_processing_semaphore, embedding_http_session, idle_watchdog_task, MARKER_ARTIFACTS

    logger.info("Application startup initiated...")

    required_vars = [GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, LIBRARY_API_TOKEN]
    if not all(required_vars):
        logger.critical("Missing required environment variables. Shutting down.")
        sys.exit(1)

    # ตั้งค่า Torch device/dtype หากยังไม่ตั้ง
    if not os.getenv("TORCH_DEVICE"):
        dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        os.environ["TORCH_DEVICE"] = dev
    if not os.getenv("TORCH_DTYPE"):
        os.environ["TORCH_DTYPE"] = "float32"
    logger.info(f"Torch device: {os.environ['TORCH_DEVICE']}, dtype: {os.environ['TORCH_DTYPE']}")

    try:
        # ตั้งค่า Gemini SDK สำหรับ Vision
        genai.configure(api_key=GEMINI_API_KEY)
        vision_model = genai.GenerativeModel('gemini-2.5-flash')  # ใช้รุ่นที่ต้องการ
        # ให้ Marker ใช้ Gemini (GOOGLE_API_KEY)
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        logger.info("Gemini Vision Model connected.")
    except Exception as e:
        logger.critical(f"Failed to connect to Gemini: {e}", exc_info=True)
        sys.exit(1)

    # โหลด Marker artifacts เพียงครั้งเดียว
    try:
        MARKER_ARTIFACTS = create_model_dict()
        logger.info("Marker artifacts loaded once and cached.")
    except Exception as e:
        logger.critical(f"Failed to load Marker artifacts: {e}", exc_info=True)
        sys.exit(1)

    # เชื่อม Qdrant (รองรับกรณีไม่มี gRPC)
    try:
        if QDRANT_USE_GRPC:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True, grpc_port=QDRANT_GRPC_PORT)
        else:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        await asyncio.to_thread(qdrant_client.get_collections)
        logger.info("Qdrant Client connected.")
    except Exception as e:
        logger.critical(f"Failed to connect to Qdrant: {e}", exc_info=True)
        sys.exit(1)

    semaphore_embedding_call = asyncio.Semaphore(CONCURRENCY)
    page_processing_semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    embedding_http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY))

    await mark_activity("startup")
    asyncio.create_task(notify_startup())
    idle_watchdog_task = asyncio.create_task(idle_watchdog())
    logger.info("Service startup complete.")
    yield
    if idle_watchdog_task:
        idle_watchdog_task.cancel()
        await asyncio.gather(idle_watchdog_task, return_exceptions=True)
    if embedding_http_session:
        await embedding_http_session.close()
    logger.info("Application shutdown.")

# ---------- FastAPI app ----------
app = FastAPI(
    title="บริการประมวลผล PDF (Marker + Multi-modal RAG)",
    description="ใช้ Marker เพื่อแปลงเอกสาร + OCR + LLM hybrid และสร้าง Index แบบ Multi-modal พร้อมบันทึกผลเป็น Markdown และรูปภาพ",
    version="5.1.0",
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
    sources = await list_unique_sources_in_collection(collection_name)
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
    return {"message": "บริการประมวลผล PDF (Marker + Multi-modal) กำลังทำงาน. ใช้ /docs สำหรับเอกสาร API."}

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
        "marker": {
            "use_llm": MARKER_USE_LLM,
            "force_ocr": MARKER_FORCE_OCR,
            "redo_inline_math": MARKER_REDO_INLINE_MATH,
            "disable_image_extraction": MARKER_DISABLE_IMAGE_EXTRACTION
        },
        "md_export": {
            "enabled": SAVE_MD_ENABLED,
            "output_dir": OCR_MD_OUTPUT_DIR
        },
        "image_export": {
            "output_dir": IMAGE_OUTPUT_DIR
        }
    }

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8999)