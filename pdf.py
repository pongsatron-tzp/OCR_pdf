import os
import asyncio
import logging
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
# Fastapi
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Path, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# External Libraries
try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False
import google.generativeai as genai
from qdrant_client import QdrantClient, models
import fitz # PyMuPDF
from PIL import Image
import io
import requests
import grpc
import httpx
import aiofiles

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
dotenv_path_key = os.path.join(os.path.dirname(__file__), 'key.env')
if os.path.exists(dotenv_path_key):
    load_dotenv(dotenv_path=dotenv_path_key)
    logger.info("โหลดตัวแปรสภาพแวดล้อมจาก key.env แล้ว")
else:
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
LIBRARY_API_TOKEN = os.getenv("LIBRARY_API_TOKEN")
LIBRARY_API_BASE_URL = "https://library-storage.agilesoftgroup.com/api"
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# --- Global Settings ---
CONCURRENCY = 6
PAGE_CONCURRENCY = 6
OCR_DPI = 600
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
QDRANT_VECTOR_SIZE = 1024
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://192.168.1.10:11434/api/embeddings")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest")
JOB_STATUS_FILE = "job_status.json"
job_status_lock = asyncio.Lock()

# --- Global Client Instances ---
vision_model: Optional[genai.GenerativeModel] = None
qdrant_client: Optional[QdrantClient] = None
semaphore_embedding_call: Optional[asyncio.Semaphore] = None
semaphore_ocr_call: Optional[asyncio.Semaphore] = None
page_processing_semaphore: Optional[asyncio.Semaphore] = None

# --- [เพิ่มฟังก์ชันกลับเข้ามา] ---
async def notify_startup():
    if not N8N_WEBHOOK_URL:
        logger.warning("[Startup] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    
    startup_webhook_url = f"{N8N_WEBHOOK_URL}_startup"
    logger.info(f"กำลังส่ง Startup Webhook notification ไปที่: {startup_webhook_url}")
    
    payload = {
        "status": "online",
        "message": "PDF Processing Service has started successfully.",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(startup_webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"ส่ง Startup Webhook notification สำเร็จ! Status: {response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อเพื่อส่ง Startup Webhook: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Startup Webhook URL ตอบกลับด้วยสถานะผิดพลาด: {e.response.status_code} - {e.response.text}")

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_model, qdrant_client, semaphore_embedding_call, semaphore_ocr_call, page_processing_semaphore
    logger.info("Application startup initiated...")
    required_vars = [GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, LIBRARY_API_TOKEN]
    if not all(required_vars):
        logger.critical("ไม่พบตัวแปร Environment ที่จำเป็น. บริการไม่สามารถเริ่มต้นได้")
        sys.exit(1)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        response = await asyncio.to_thread(vision_model.generate_content, "test vision model connectivity")
        if not response.text: raise Exception("Gemini Vision model test returned no text.")
        logger.info("โหลดและทดสอบ Gemini Vision Model ('gemini-2.5-flash') สำเร็จแล้ว")
    except Exception as e:
        logger.critical(f"โหลดหรือทดสอบ Gemini Vision Model ล้มเหลว: {e}", exc_info=True)
        sys.exit(1)
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
        await asyncio.to_thread(qdrant_client.get_collections)
        logger.info("เชื่อมต่อกับ Qdrant Cloud Client สำเร็จแล้ว")
    except Exception as e:
        logger.critical(f"เชื่อมต่อกับ Qdrant Cloud Client ล้มเหลว: {e}", exc_info=True)
        sys.exit(1)
    semaphore_embedding_call = asyncio.Semaphore(CONCURRENCY)
    semaphore_ocr_call = asyncio.Semaphore(CONCURRENCY)
    page_processing_semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    logger.info(f"การเริ่มต้นบริการเสร็จสมบูรณ์. API Concurrency: {CONCURRENCY}, Page Concurrency: {PAGE_CONCURRENCY}.")
    
    tasks = BackgroundTasks()
    tasks.add_task(notify_startup)
    await tasks()
    
    yield
    
    logger.info("Application shutdown.")

# --- FastAPI Application ---
app = FastAPI(
    title="บริการประมวลผล PDF (Asynchronous with Job Tracking)",
    description="รับไฟล์, คืน Path สำหรับติดตาม, ประมวลผลในเบื้องหลัง, และสามารถตรวจสอบสถานะได้",
    version="2.3.1",
    lifespan=lifespan
)

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

# --- Helper Functions for Job Status ---
async def _read_job_statuses() -> Dict[str, Any]:
    async with job_status_lock:
        if not os.path.exists(JOB_STATUS_FILE):
            return {}
        async with aiofiles.open(JOB_STATUS_FILE, mode='r') as f:
            content = await f.read()
            if not content: return {}
            return json.loads(content)

async def _write_job_statuses(statuses: Dict[str, Any]):
    async with job_status_lock:
        async with aiofiles.open(JOB_STATUS_FILE, mode='w') as f:
            await f.write(json.dumps(statuses, indent=2))

async def update_job_status(file_path: str, status: str, details: Optional[Dict[str, Any]] = None):
    statuses = await _read_job_statuses()
    now_utc = datetime.now(timezone.utc).isoformat()
    if file_path not in statuses:
        statuses[file_path] = {"created_at": now_utc}
    statuses[file_path]["status"] = status
    statuses[file_path]["updated_at"] = now_utc
    if details:
        statuses[file_path].update(details)
    await _write_job_statuses(statuses)

# --- [เพิ่มฟังก์ชันกลับเข้ามา] ---
async def notify_webhook(result_data: ProcessingResponse):
    if not N8N_WEBHOOK_URL:
        logger.warning("[BG] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    logger.info(f"[BG] กำลังส่ง Webhook notification ไปที่: {N8N_WEBHOOK_URL}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_WEBHOOK_URL, json=json.loads(result_data.model_dump_json()), timeout=30)
            response.raise_for_status()
            logger.info(f"[BG] ส่ง Webhook notification สำเร็จ! Status: {response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"[BG] เกิดข้อผิดพลาดในการเชื่อมต่อเพื่อส่ง Webhook: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"[BG] Webhook URL ตอบกลับด้วยสถานะผิดพลาด: {e.response.status_code} - {e.response.text}")

# --- Other Helper Functions ---
def parse_page_string(page_str: Optional[str]) -> Set[int]:
    if not page_str: return set()
    page_numbers = set()
    parts = page_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end: start, end = end, start
                page_numbers.update(range(start, end + 1))
            else:
                page_numbers.add(int(part))
        except ValueError:
            logger.warning(f"ไม่สามารถแปลงค่าหน้า '{part}' ได้ จะข้ามส่วนนี้ไป")
    return page_numbers

async def get_existing_page_numbers(collection_name: str, source_file_name: str) -> Set[int]:
    global qdrant_client
    if qdrant_client is None: return set()
    existing_pages = set()
    try:
        source_filter = models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source_file_name))])
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(qdrant_client.scroll, collection_name=collection_name, scroll_filter=source_filter, limit=250, offset=next_offset, with_payload=["metadata.loc.pageNumber"], with_vectors=False)
            for point in response:
                page_num = point.payload.get("metadata", {}).get("loc", {}).get("pageNumber")
                if page_num is not None:
                    existing_pages.add(page_num)
            if next_offset is None:
                break
        return existing_pages
    except Exception as e:
        logger.warning(f"เกิดข้อผิดพลาดขณะดึงข้อมูลหน้าที่มีอยู่: {e}. จะถือว่ายังไม่มีหน้าใดๆ")
        return set()

async def list_unique_sources_in_collection(collection_name: str) -> List[str]:
    global qdrant_client
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client is not available.")
    unique_sources = set()
    try:
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(qdrant_client.scroll, collection_name=collection_name, limit=250, offset=next_offset, with_payload=["metadata.source"], with_vectors=False)
            for point in response:
                source = point.payload.get("metadata", {}).get("source")
                if source:
                    unique_sources.add(source)
            if next_offset is None:
                break
        return sorted(list(unique_sources))
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดขณะ list sources จาก collection '{collection_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or an error occurred.")

async def search_library_for_book(query: str) -> Optional[List[Dict[str, Any]]]:
    search_url = f"{LIBRARY_API_BASE_URL}/search"
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}
    params = {"q": query}
    logger.info(f"กำลังค้นหาหนังสือใน Library ด้วยคำว่า: '{query}'...")
    try:
        response = await asyncio.to_thread(requests.get, search_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        if isinstance(response_data, dict) and 'files' in response_data:
            return response_data['files']
        logger.warning(f"ไม่พบไฟล์ใน Library ที่ตรงกับคำค้นหา: '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อเพื่อค้นหาหนังสือใน Library: {e}")
        return None

async def download_book_from_library(file_path: str) -> Optional[Tuple[str, bytes]]:
    filename = os.path.basename(file_path)
    download_url = f"{LIBRARY_API_BASE_URL}/files/download/{file_path}"
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}
    logger.info(f"กำลังเริ่มดาวน์โหลดไฟล์ '{filename}' จาก Library...")
    def _blocking_download_with_progress():
        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                file_in_memory = io.BytesIO()
                if total_size_in_bytes == 0:
                    logger.warning("ไม่พบข้อมูล Content-Length, ไม่สามารถแสดงความคืบหน้าเป็น % ได้")
                    for chunk in r.iter_content(chunk_size=8192):
                        file_in_memory.write(chunk)
                else:
                    downloaded_size = 0
                    last_logged_percent = -10
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_size += len(chunk)
                            file_in_memory.write(chunk)
                            current_percent = int((downloaded_size / total_size_in_bytes) * 100)
                            if current_percent >= last_logged_percent + 10:
                                logger.info(f"Downloading '{filename}'... {current_percent}%")
                                last_logged_percent = current_percent
                logger.info(f"ดาวน์โหลดไฟล์ '{filename}' จาก Library สำเร็จ (100%)")
                return file_in_memory.getvalue()
        except requests.exceptions.RequestException as e:
            logger.error(f"เกิดข้อผิดพลาดระหว่างดาวน์โหลดไฟล์ '{filename}' จาก Library: {e}")
            return None
    file_bytes = await asyncio.to_thread(_blocking_download_with_progress)
    if file_bytes is not None:
        return filename, file_bytes
    else:
        return None

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text): break
        start += (chunk_size - chunk_overlap)
        start = max(0, start)
    return chunks

async def async_generate_embedding(input_data: List[str]) -> Optional[List[List[float]]]:
    if not input_data: return None
    url = OLLAMA_EMBEDDING_URL
    headers = {"Content-Type": "application/json"}
    embeddings = []
    for text_to_embed in input_data:
        payload = {"model": OLLAMA_EMBEDDING_MODEL_NAME, "prompt": text_to_embed}
        try:
            response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            resp_json = response.json()
            if "embedding" in resp_json and isinstance(resp_json["embedding"], list):
                embeddings.append(resp_json["embedding"])
            else:
                embeddings.append(None)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการรับ Embedding: {e}", exc_info=True)
            embeddings.append(None)
    valid_embeddings = [e for e in embeddings if e is not None]
    return valid_embeddings if valid_embeddings else None

async def async_process_text_chunk(chunk_text: str, chunk_metadata: Dict[str, Any]) -> models.PointStruct | None:
    global semaphore_embedding_call
    if semaphore_embedding_call is None: return None
    async with semaphore_embedding_call:
        chunk_id = chunk_metadata.get('chunk_id')
        if not chunk_id:
            logger.error("ไม่พบ chunk_id (UUID) ใน metadata")
            return None
        embedding_results = await async_generate_embedding([chunk_text])
        if not (embedding_results and embedding_results[0]):
            logger.warning(f"สร้าง embedding สำหรับ chunk '{chunk_id}' ล้มเหลว")
            return None
        payload = {"pageContent": chunk_text, "metadata": chunk_metadata}
        return models.PointStruct(id=chunk_id, vector=embedding_results[0], payload=payload)

async def ensure_qdrant_collection(collection_name: str):
    global qdrant_client
    if qdrant_client is None: raise RuntimeError("Qdrant client not initialized")
    try:
        await asyncio.to_thread(qdrant_client.get_collection, collection_name=collection_name)
    except Exception:
        # ถ้าไม่มี ให้สร้างขึ้นมาใหม่
        logger.info(f"ไม่พบ Collection '{collection_name}' กำลังสร้าง...")
        await asyncio.to_thread(
            qdrant_client.create_collection,
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=QDRANT_VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' สร้างสำเร็จแล้ว.")

        # สร้าง Index สำหรับ source (เหมือนเดิม)
        logger.info(f"กำลังสร้าง Payload Index (Keyword) สำหรับ 'metadata.source'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        # 2. Index สำหรับ pageNumber
        logger.info(f"กำลังสร้าง Payload Index (Integer) สำหรับ 'metadata.loc.pageNumber'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.loc.pageNumber",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        
        # 3. Index สำหรับ chunkIndex
        logger.info(f"กำลังสร้าง Payload Index (Integer) สำหรับ 'metadata.loc.chunkIndex'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.loc.chunkIndex",
            field_schema=models.PayloadSchemaType.INTEGER
        )

        # สร้าง Index สำหรับ pageContent (Full-text search)
        logger.info(f"กำลังสร้าง Payload Index (Full-text) สำหรับ 'pageContent'...")
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
        logger.info(f"สร้าง Payload Index ทั้งหมดสำหรับ collection '{collection_name}' สำเร็จแล้ว.")


async def process_and_upsert_single_page(doc: fitz.Document, page_num: int, file_name: str, collection_name: str) -> Tuple[int, int]:
    global vision_model, semaphore_ocr_call, qdrant_client
    page_index = page_num - 1
    logger.info(f"--- [BG] เริ่มประมวลผลหน้า {page_num} ---")
    page_tasks = []
    try:
        page = doc.load_page(page_index)
        page_text = await asyncio.to_thread(page.get_text)
        if len(page_text.strip()) < 50 and len(page.get_text("words")) < 10:
            logger.info(f"[BG] กำลังลอง Gemini Vision OCR สำหรับหน้า {page_num}...")
            pix = await asyncio.to_thread(page.get_pixmap, dpi=OCR_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_prompt = [img, "Extract all text from this image. Return the text exactly as it appears."]
            try:
                async with semaphore_ocr_call:
                    ocr_response = await asyncio.to_thread(vision_model.generate_content, ocr_prompt)
                    ocr_response.resolve()
                    page_text = ocr_response.text
            except Exception as e:
                logger.error(f"[BG] Gemini Vision OCR ล้มเหลวสำหรับหน้า {page_num}: {e}", exc_info=True)
                page_text = ""
        if not page_text.strip():
            logger.info(f"--- [BG] หน้า {page_num} ไม่มีข้อความให้ประมวลผล ---")
            return 0, 0
        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_index_on_page, chunk in enumerate(page_chunks):
            chunk_metadata = {"source": file_name, "loc": {"pageNumber": page_num, "chunkIndex": chunk_index_on_page}, "chunk_id": str(uuid.uuid4())}
            task = asyncio.create_task(async_process_text_chunk(chunk, chunk_metadata))
            page_tasks.append(task)
        if not page_tasks:
            logger.info(f"--- [BG] หน้า {page_num} ไม่ได้สร้าง chunks ---")
            return 0, 0
        results = await asyncio.gather(*page_tasks, return_exceptions=True)
        successful_points = [r for r in results if r is not None and not isinstance(r, Exception)]
        failed_count = len(page_tasks) - len(successful_points)
        if successful_points:
            await asyncio.to_thread(qdrant_client.upsert, collection_name=collection_name, wait=True, points=successful_points)
            logger.info(f"[BG] บันทึก {len(successful_points)} chunks จากหน้า {page_num} ลง Qdrant สำเร็จ")
        logger.info(f"--- [BG] จบการประมวลผลหน้า {page_num} ---")
        return len(successful_points), failed_count
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดรุนแรงขณะประมวลผลหน้า {page_num}: {e}", exc_info=True)
        return 0, len(page_tasks)

async def page_worker_with_semaphore(doc: fitz.Document, page_num: int, file_name: str, collection_name: str) -> Tuple[int, int]:
    global page_processing_semaphore
    if page_processing_semaphore is None:
        raise RuntimeError("Page processing semaphore is not initialized.")
    async with page_processing_semaphore:
        return await process_and_upsert_single_page(doc, page_num, file_name, collection_name)

async def process_pdf_in_background(
    file_path_key: str,
    file_bytes: bytes,
    file_name: str,
    pages_str: Optional[str] = None,
    custom_collection_name: Optional[str] = None
):
    original_file_name = file_name.strip()
    cleaned_source_name = os.path.splitext(original_file_name)[0]
    
    if custom_collection_name:
        collection_name = custom_collection_name.strip().lower()
    else:
        collection_name = "".join(c for c in cleaned_source_name if c.isalnum() or c in ['-', '_']).lower()
        if not collection_name:
            collection_name = f"pdf_doc_{uuid.uuid4().hex}"
    
    await update_job_status(file_path_key, "processing", {"file_path": original_file_name, "collection_name": collection_name})
    final_response = None
    doc = None
    try:
        await ensure_qdrant_collection(collection_name)
        requested_pages = parse_page_string(pages_str)
        existing_pages = await get_existing_page_numbers(collection_name, cleaned_source_name)
        if existing_pages:
            logger.info(f"[BG] พบหน้าที่ประมวลผลแล้วสำหรับไฟล์นี้: {sorted(list(existing_pages))}")

        doc = await asyncio.to_thread(fitz.open, stream=file_bytes, filetype="pdf")
        total_pages_in_doc = len(doc)
        
        if requested_pages:
            pages_to_process = requested_pages - existing_pages
        else:
            all_doc_pages = set(range(1, total_pages_in_doc + 1))
            pages_to_process = all_doc_pages - existing_pages
            
        if not pages_to_process:
            logger.warning(f"[BG] หน้าที่ร้องขอทั้งหมดสำหรับไฟล์ '{cleaned_source_name}' มีอยู่แล้วใน collection. ข้ามการประมวลผล")
            final_response = ProcessingResponse(collection_name=collection_name, status="skipped", processed_chunks=0, failed_chunks=0, message="หน้าที่ร้องขอทั้งหมดมีอยู่แล้วใน collection", file_name=original_file_name)
        else:
            pages_to_iterate = sorted([p for p in pages_to_process if 1 <= p <= total_pages_in_doc])
            logger.info(f"[BG] จะทำการประมวลผลหน้าใหม่ {len(pages_to_iterate)} หน้า (สูงสุด {PAGE_CONCURRENCY} หน้าพร้อมกัน): {pages_to_iterate}")
            page_processing_tasks = []
            for page_num in pages_to_iterate:
                task = asyncio.create_task(page_worker_with_semaphore(doc, page_num, cleaned_source_name, collection_name))
                page_processing_tasks.append(task)
            results = await asyncio.gather(*page_processing_tasks)
            doc.close()
            doc = None

            total_successful_chunks = sum(res[0] for res in results)
            total_failed_chunks = sum(res[1] for res in results)

            if total_successful_chunks > 0:
                message = f"ประมวลผลและเพิ่ม {total_successful_chunks} chunks ใหม่ ลงใน Qdrant collection '{collection_name}' สำเร็จแล้ว"
                status_code = "success"
            else:
                message = f"ไม่มี chunks ใหม่ใดๆ ถูกประมวลผลสำเร็จสำหรับ {original_file_name}."
                status_code = "warning" if total_failed_chunks == 0 else "error"
            final_response = ProcessingResponse(collection_name=collection_name, status=status_code, processed_chunks=total_successful_chunks, failed_chunks=total_failed_chunks, message=message, file_name=original_file_name)
        
        await update_job_status(file_path_key, "completed", {"result": json.loads(final_response.model_dump_json())})

    except Exception as e:
        logger.error(f"[BG Task] เกิดข้อผิดพลาดรุนแรงที่ไม่สามารถจัดการได้: {e}", exc_info=True)
        error_message = f"เกิดข้อผิดพลาดรุนแรงระหว่างการประมวลผล: {e}"
        final_response = ProcessingResponse(collection_name=collection_name, status="failed", processed_chunks=0, failed_chunks=0, message=error_message, file_name=original_file_name)
        await update_job_status(file_path_key, "failed", {"error": error_message})
    finally:
        if doc is not None:
            doc.close()
            logger.info(f"[BG] ปิดเอกสาร '{original_file_name}' เรียบร้อยแล้ว")
        if final_response:
            await notify_webhook(final_response)

# --- FastAPI Endpoints ---
@app.post("/process_pdf/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_pdf_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), pages: Optional[str] = Form(None), collection_name: Optional[str] = Form(None)):
    file_path_key = file.filename
    logger.info(f"ได้รับไฟล์: {file_path_key}, กำลังเพิ่ม Task เข้าสู่เบื้องหลัง...")
    file_bytes = await file.read()
    
    await update_job_status(file_path_key, "queued", {"file_path": file_path_key})
    
    background_tasks.add_task(process_pdf_in_background, file_path_key=file_path_key, file_bytes=file_bytes, file_name=file.filename, pages_str=pages, custom_collection_name=collection_name)
    
    return AcknowledgementResponse(message="Task accepted. Use the file_path to check status.", file_path=file_path_key, task_status="queued")

@app.get("/collections/{collection_name}/sources", response_model=SourceListResponse)
async def get_sources_in_collection(collection_name: str = Path(..., description="ชื่อของ Collection ที่ต้องการตรวจสอบ")):
    logger.info(f"ได้รับคำขอเพื่อ list sources ใน collection: '{collection_name}'")
    sources = await list_unique_sources_in_collection(collection_name)
    return SourceListResponse(collection_name=collection_name, source_count=len(sources), sources=sources)

@app.post("/process_from_library/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_from_library(request: LibrarySearchRequest, background_tasks: BackgroundTasks):
    logger.info(f"ได้รับคำขอจาก Library: '{request.query}', กำลังดาวน์โหลด...")
    found_files = await search_library_for_book(request.query)
    if not found_files:
        raise HTTPException(status_code=404, detail=f"ไม่พบหนังสือใน Library: '{request.query}'")
    
    file_to_process = found_files[0]
    file_path_key = file_to_process.get('path')
    if not file_path_key:
        raise HTTPException(status_code=500, detail="ข้อมูลจาก Library API ไม่มี 'path' ของไฟล์")

    download_result = await download_book_from_library(file_path_key)
    if download_result is None:
        raise HTTPException(status_code=500, detail=f"ดาวน์โหลดไฟล์ '{file_path_key}' จาก Library ล้มเหลว")

    file_name, file_bytes = download_result
    
    await update_job_status(file_path_key, "queued", {"file_path": file_name})
    
    background_tasks.add_task(process_pdf_in_background, file_path_key=file_path_key, file_bytes=file_bytes, file_name=file_name, pages_str=request.pages, custom_collection_name=request.collection_name)

    return AcknowledgementResponse(message="Task accepted. Use the file_path to check status.", file_path=file_path_key, task_status="queued")

@app.post("/process_by_path/", response_model=AcknowledgementResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_by_file_path(request: ProcessByPathRequest, background_tasks: BackgroundTasks):
    file_path_key = request.file_path
    logger.info(f"ได้รับคำขอจาก Path: '{file_path_key}', กำลังเพิ่ม Task...")
    
    download_result = await download_book_from_library(file_path_key)
    if download_result is None:
        raise HTTPException(status_code=500, detail=f"ดาวน์โหลดไฟล์ '{file_path_key}' จาก Library ล้มเหลว")

    file_name, file_bytes = download_result
    
    await update_job_status(file_path_key, "queued", {"file_path": file_name})
    
    background_tasks.add_task(process_pdf_in_background, file_path_key=file_path_key, file_bytes=file_bytes, file_name=file_name, pages_str=request.pages, custom_collection_name=request.collection_name)

    return AcknowledgementResponse(message="Task accepted. Use the file_path to check status.", file_path=file_path_key, task_status="queued")

@app.get("/status", response_model=JobStatusResponse)
async def get_job_status(file_path: str = Query(..., description="File Path ที่ได้รับจากการส่งไฟล์ (ต้อง URL Encoded)")):
    statuses = await _read_job_statuses()
    job_details = statuses.get(file_path)
    if not job_details:
        raise HTTPException(status_code=404, detail=f"Status for file_path '{file_path}' not found.")
    return JobStatusResponse(file_path=file_path, details=job_details)

@app.get("/")
async def root():
    return {"message": "บริการประมวลผล PDF กำลังทำงาน. ใช้ /docs สำหรับเอกสาร API."}

# --- Main Execution ---
if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("!!!! โปรดใช้ Python 3.8+ !!!!")
        sys.exit(1)
    if not all([GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, LIBRARY_API_TOKEN]):
        logger.critical("ไม่พบตัวแปรสภาพแวดล้อมที่จำเป็น. โปรดตรวจสอบ key.env")
        sys.exit(1)
    import uvicorn
    uvicorn.run("pdf:app", host="0.0.0.0", port=8080, reload=True)