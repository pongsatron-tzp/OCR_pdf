import os
import asyncio
import logging
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from enum import Enum
from dataclasses import dataclass
import queue

# FastAPI
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
import fitz  # PyMuPDF
from PIL import Image
import io
import requests
import grpc
import httpx
import aiofiles
from urllib.parse import urljoin, quote
import aiohttp  # <- Added

# Typhoon OCR (แทน HTTP API)
try:
    from typhoon_ocr import ocr_document
    _typhoon_ocr_available = True
except ImportError:
    _typhoon_ocr_available = False
    logging.warning("typhoon-ocr package ไม่ได้ติดตั้ง, จะใช้เฉพาะ Gemini")

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

# API Keys และ URLs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
LIBRARY_API_TOKEN = os.getenv("LIBRARY_API_TOKEN")
LIBRARY_API_BASE_URL = "https://library-storage.agilesoftgroup.com"
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# --- Global Settings ---
CONCURRENCY = 30
PAGE_CONCURRENCY = 30
OCR_DPI = 600
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
QDRANT_VECTOR_SIZE = 1024

# Worker Pool Settings
TYPHOON_WORKERS = 20  # จำนวน workers สำหรับ Typhoon
GEMINI_WORKERS = 10   # จำนวน workers สำหรับ Gemini

OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "https://jetson-embleding.agilesoftgroup.com/api/embeddings")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest")

JOB_STATUS_FILE = "job_status.json"
job_status_lock = asyncio.Lock()

# --- OCR System Classes ---
class OCRProvider(Enum):
    TYPHOON = "typhoon"
    GEMINI = "gemini"

@dataclass
class OCRTask:
    """OCR Task สำหรับใส่ใน Queue"""
    task_id: str
    page_num: int
    image: Image.Image
    file_name: str
    collection_name: str
    retry_count: int = 0
    max_retries: int = 3
    preferred_provider: Optional[OCRProvider] = None
    temp_image_path: Optional[str] = None

@dataclass
class OCRResult:
    """ผลลัพธ์จาก OCR"""
    task_id: str
    success: bool
    text: str = ""
    provider_used: Optional[OCRProvider] = None
    error: Optional[str] = None

class DualOCRManager:
    """ตัวจัดการ OCR แบบ Dual API พร้อม Dynamic Dispatch"""
    
    def __init__(self):
        # แยกคิวตามผู้ให้บริการ
        self.typhoon_queue: asyncio.Queue[OCRTask] = asyncio.Queue()
        self.gemini_queue: asyncio.Queue[OCRTask] = asyncio.Queue()
        # Futures ต่อ task_id
        self.pending: Dict[str, asyncio.Future] = {}
        
        # Worker pools
        self.typhoon_workers: List[asyncio.Task] = []
        self.gemini_workers: List[asyncio.Task] = []
        
        # Counters สำหรับสถิติ
        self.typhoon_success_count = 0
        self.typhoon_error_count = 0
        self.gemini_success_count = 0
        self.gemini_error_count = 0
        
        # Models
        self.vision_model: Optional[genai.GenerativeModel] = None
        
        self.running = False
        
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_ocr_images")
        os.makedirs(self.temp_dir, exist_ok=True)

    async def _enqueue_by_preference(self, task: OCRTask):
        """เลือกคิวตาม preferred_provider โดยเคารพความพร้อมของ provider"""
        if task.preferred_provider == OCRProvider.GEMINI:
            await self.gemini_queue.put(task)
            return
        if task.preferred_provider == OCRProvider.TYPHOON:
            if _typhoon_ocr_available:
                await self.typhoon_queue.put(task)
            else:
                await self.gemini_queue.put(task)
            return
        # default: Typhoon ก่อนถ้าพร้อม
        if _typhoon_ocr_available:
            await self.typhoon_queue.put(task)
        else:
            await self.gemini_queue.put(task)
        
    async def initialize(self):
        """เริ่มต้น OCR Manager"""
        global _typhoon_ocr_available  # ย้ายมาไว้บนสุด
        # Initialize Gemini
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
            # Test connection
            response = await asyncio.to_thread(
                self.vision_model.generate_content, 
                "test vision model connectivity"
            )
            if not response.text:
                raise Exception("Gemini Vision model test returned no text.")
            logger.info("เชื่อมต่อ Gemini Vision Model สำเร็จ")
        except Exception as e:
            logger.error(f"เชื่อมต่อ Gemini Vision Model ล้มเหลว: {e}")
            raise

        # Initialize Typhoon OCR
        if _typhoon_ocr_available:
            try:
                # Set API key for typhoon-ocr
                if TYPHOON_API_KEY:
                    os.environ["TYPHOON_OCR_API_KEY"] = TYPHOON_API_KEY

                async def _test_typhoon_ocr():  # เอา self ออก
                    """ทดสอบ Typhoon OCR Package"""
                    # สร้างภาพทดสอบขนาดเล็ก
                    test_image = Image.new('RGB', (100, 100), color='white')
                    temp_path = os.path.join(self.temp_dir, f"test_{uuid.uuid4().hex}.png")
                    try:
                        test_image.save(temp_path)
                        # ทดสอบ typhoon-ocr
                        result = await asyncio.to_thread(
                            ocr_document,
                            pdf_or_image_path=temp_path,
                            task_type="default"
                        )
                        if not isinstance(result, str):
                            raise Exception(f"Unexpected result type: {type(result)}")
                        logger.info("Typhoon OCR test สำเร็จ")
                    finally:
                        # ลบไฟล์ทดสอบ
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                await _test_typhoon_ocr()  # เอา self ออก
                logger.info("Typhoon OCR Package พร้อมใช้งาน")
            except Exception as e:
                logger.error(f"Typhoon OCR Package เกิดข้อผิดพลาด: {e}")
                # ไม่หยุดการทำงาน แค่ใช้ Gemini อย่างเดียว
                _typhoon_ocr_available = False
        else:
            logger.warning("Typhoon OCR Package ไม่พร้อมใช้งาน, จะใช้เฉพาะ Gemini")
            
    async def start_workers(self):
        """เริ่ม Worker pools"""
        if self.running:
            return
            
        self.running = True
        
        # Start Typhoon workers
        if _typhoon_ocr_available:
            for i in range(TYPHOON_WORKERS):
                task = asyncio.create_task(self._typhoon_worker(f"typhoon-{i}"))
                self.typhoon_workers.append(task)
            
        # Start Gemini workers  
        for i in range(GEMINI_WORKERS):
            task = asyncio.create_task(self._gemini_worker(f"gemini-{i}"))
            self.gemini_workers.append(task)
            
        typhoon_count = len(self.typhoon_workers)
        gemini_count = len(self.gemini_workers)
        logger.info(f"เริ่ม OCR Workers: {typhoon_count} Typhoon, {gemini_count} Gemini")
        
    async def stop_workers(self):
        """หยุด Worker pools"""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all workers
        all_workers = self.typhoon_workers + self.gemini_workers
        for worker in all_workers:
            worker.cancel()
            
        # Wait for cancellation
        await asyncio.gather(*all_workers, return_exceptions=True)
        
        # Clean up temp files
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"ไม่สามารถลบโฟลเดอร์ temp: {e}")
            
        logger.info("หยุด OCR Workers แล้ว")

    async def _finalize_failure(self, task: OCRTask, provider: OCRProvider, error_msg: str):
        """สรุปผลล้มเหลวให้ future (กรณี exception นอก flow)"""
        result = OCRResult(
            task_id=task.task_id,
            success=False,
            error=error_msg,
            provider_used=provider
        )
        fut = self.pending.pop(task.task_id, None)
        if fut and not fut.done():
            fut.set_result(result)
        
    async def _typhoon_worker(self, worker_id: str):
        """Typhoon OCR Worker"""
        logger.info(f"เริ่ม Typhoon Worker: {worker_id}")
        
        while self.running:
            try:
                # รอ task จากคิวของ Typhoon
                task = await self.typhoon_queue.get()
                
                logger.info(f"[{worker_id}] รับ task {task.task_id} (หน้า {task.page_num})")
                
                # ทำ OCR ด้วย Typhoon
                result = await self._perform_typhoon_ocr(task, worker_id)
                
                # ถ้าได้ผลลัพธ์สุดท้ายแล้ว ให้ set_result
                if result is not None:
                    fut = self.pending.pop(task.task_id, None)
                    if fut and not fut.done():
                        fut.set_result(result)
                
                # Mark task done
                self.typhoon_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{worker_id}] เกิดข้อผิดพลาด: {e}")
                # finalize failure
                if 'task' in locals():
                    await self._finalize_failure(task, OCRProvider.TYPHOON, str(e))
                    self.typhoon_queue.task_done()
                
    async def _gemini_worker(self, worker_id: str):
        """Gemini OCR Worker"""
        logger.info(f"เริ่ม Gemini Worker: {worker_id}")
        
        while self.running:
            try:
                # รอ task จากคิวของ Gemini
                task = await self.gemini_queue.get()
                
                logger.info(f"[{worker_id}] รับ task {task.task_id} (หน้า {task.page_num})")
                
                # ทำ OCR ด้วย Gemini
                result = await self._perform_gemini_ocr(task, worker_id)
                
                # ถ้าได้ผลลัพธ์สุดท้ายแล้ว ให้ set_result
                if result is not None:
                    fut = self.pending.pop(task.task_id, None)
                    if fut and not fut.done():
                        fut.set_result(result)
                
                # Mark task done
                self.gemini_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{worker_id}] เกิดข้อผิดพลาด: {e}")
                # finalize failure
                if 'task' in locals():
                    await self._finalize_failure(task, OCRProvider.GEMINI, str(e))
                    self.gemini_queue.task_done()
                
    async def _perform_typhoon_ocr(self, task: OCRTask, worker_id: str) -> Optional[OCRResult]:
        """ทำ OCR ด้วย Typhoon OCR Package"""
        try:
            # บันทึกรูปภาพเป็นไฟล์ชั่วคราว
            temp_path = os.path.join(self.temp_dir, f"{task.task_id}.png")
            task.image.save(temp_path)
            task.temp_image_path = temp_path

            # เรียกใช้ typhoon-ocr
            extracted_text = await asyncio.to_thread(
                ocr_document,
                pdf_or_image_path=temp_path,
                task_type="default"
            )
            self.typhoon_success_count += 1
            logger.info(f"[{worker_id}] Typhoon OCR สำเร็จ สำหรับ {task.task_id}")
            return OCRResult(
                task_id=task.task_id,
                success=True,
                text=extracted_text,
                provider_used=OCRProvider.TYPHOON
            )
        except Exception as e:
            self.typhoon_error_count += 1
            logger.error(f"[{worker_id}] Typhoon OCR ล้มเหลว {task.task_id}: {e}")
            # ถ้ายังทำ retry ได้ ให้ส่งไปคิว Gemini
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.preferred_provider = OCRProvider.GEMINI  # ลอง Gemini แทน
                await self.gemini_queue.put(task)
                logger.info(f"[{worker_id}] ส่ง {task.task_id} ไป retry ด้วย Gemini (retry={task.retry_count})")
                return None  # ยังไม่ finalize
            return OCRResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                provider_used=OCRProvider.TYPHOON
            )
        finally:
            # ลบไฟล์ชั่วคราว
            if task.temp_image_path and os.path.exists(task.temp_image_path):
                try:
                    os.remove(task.temp_image_path)
                except Exception as e:
                    logger.warning(f"ไม่สามารถลบไฟล์ temp {task.temp_image_path}: {e}")
            
    async def _perform_gemini_ocr(self, task: OCRTask, worker_id: str) -> Optional[OCRResult]:
        """ทำ OCR ด้วย Gemini Vision"""
        try:
            ocr_prompt = [
                task.image,
                "Extract all text from this image. Return the text exactly as it appears."
            ]
            
            response = await asyncio.to_thread(
                self.vision_model.generate_content, 
                ocr_prompt
            )
            response.resolve()
            extracted_text = response.text
            
            self.gemini_success_count += 1
            logger.info(f"[{worker_id}] Gemini OCR สำเร็จ สำหรับ {task.task_id}")
            
            return OCRResult(
                task_id=task.task_id,
                success=True,
                text=extracted_text,
                provider_used=OCRProvider.GEMINI
            )
            
        except Exception as e:
            self.gemini_error_count += 1
            logger.error(f"[{worker_id}] Gemini OCR ล้มเหลว {task.task_id}: {e}")
            
            # ถ้ายังทำ retry ได้ ให้ส่งไปคิว Typhoon (ถ้ามี)
            if task.retry_count < task.max_retries and _typhoon_ocr_available:
                task.retry_count += 1
                task.preferred_provider = OCRProvider.TYPHOON  # ลอง Typhoon แทน
                await self.typhoon_queue.put(task)
                logger.info(f"[{worker_id}] ส่ง {task.task_id} ไป retry ด้วย Typhoon (retry={task.retry_count})")
                return None  # ยังไม่ finalize
            
            return OCRResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                provider_used=OCRProvider.GEMINI
            )
            
    async def submit_ocr_task(self, task: OCRTask) -> OCRResult:
        """เพิ่ม OCR task เข้าคิวตาม provider และรอผลของ task นั้น"""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending[task.task_id] = fut
        await self._enqueue_by_preference(task)
        return await fut
        
    def get_stats(self) -> Dict[str, Any]:
        """ได้สถิติการใช้งาน"""
        return {
            "typhoon": {
                "success": self.typhoon_success_count,
                "error": self.typhoon_error_count,
                "active_workers": len([w for w in self.typhoon_workers if not w.done()]),
                "available": _typhoon_ocr_available
            },
            "gemini": {
                "success": self.gemini_success_count,
                "error": self.gemini_error_count,
                "active_workers": len([w for w in self.gemini_workers if not w.done()])
            },
            "queue_size": self.typhoon_queue.qsize() + self.gemini_queue.qsize()
        }

# --- Global Client Instances ---
vision_model: Optional[genai.GenerativeModel] = None
qdrant_client: Optional[QdrantClient] = None
semaphore_embedding_call: Optional[asyncio.Semaphore] = None
page_processing_semaphore: Optional[asyncio.Semaphore] = None

# OCR Manager Instance
ocr_manager: Optional[DualOCRManager] = None

# Embedding HTTP session (aiohttp)
embedding_http_session: Optional[aiohttp.ClientSession] = None  # <- Added

# --- เพิ่มฟังก์ชันกลับเข้ามา ---
async def notify_startup():
    if not N8N_WEBHOOK_URL:
        logger.warning("[Startup] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    
    startup_webhook_url = f"{N8N_WEBHOOK_URL}"
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
    global vision_model, qdrant_client, semaphore_embedding_call, page_processing_semaphore, ocr_manager, embedding_http_session
    
    logger.info("Application startup initiated...")
    
    required_vars = [GEMINI_API_KEY, TYPHOON_API_KEY, QDRANT_URL, QDRANT_API_KEY, LIBRARY_API_TOKEN]
    if not all(required_vars):
        logger.critical("ไม่พบตัวแปร Environment ที่จำเป็น. บริการไม่สามารถเริ่มต้นได้")
        sys.exit(1)

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True, grpc_port=6334)
        await asyncio.to_thread(qdrant_client.get_collections)
        logger.info("เชื่อมต่อกับ Qdrant Cloud Client สำเร็จแล้ว")
    except Exception as e:
        logger.critical(f"เชื่อมต่อกับ Qdrant Cloud Client ล้มเหลว: {e}", exc_info=True)
        sys.exit(1)
        
    try:
        # Initialize OCR Manager
        ocr_manager = DualOCRManager()
        await ocr_manager.initialize()
        await ocr_manager.start_workers()
        logger.info("OCR Manager เริ่มต้นสำเร็จ")
        
    except Exception as e:
        logger.critical(f"OCR Manager เริ่มต้นล้มเหลว: {e}", exc_info=True)
        sys.exit(1)
    
    semaphore_embedding_call = asyncio.Semaphore(CONCURRENCY)
    page_processing_semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)

    # Create shared aiohttp session for embeddings
    embedding_http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=CONCURRENCY)
    )
    
    logger.info(f"การเริ่มต้นบริการเสร็จสมบูรณ์. API Concurrency: {CONCURRENCY}, Page Concurrency: {PAGE_CONCURRENCY}.")
    
    tasks = BackgroundTasks()
    tasks.add_task(notify_startup)
    await tasks()
    
    yield
    
    # Cleanup
    if ocr_manager:
        await ocr_manager.stop_workers()
    if embedding_http_session:
        await embedding_http_session.close()
    logger.info("Application shutdown.")

# --- FastAPI Application ---
app = FastAPI(
    title="บริการประมวลผล PDF (Dual OCR with Dynamic Dispatch)",
    description="ใช้ Typhoon + Gemini APIs พร้อมกันเพื่อ OCR ที่เร็วขึ้น",
    version="3.0.0",
    lifespan=lifespan
)

# --- Pydantic Models (เดิม + เพิ่ม OCR Stats) ---
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

class OCRStatsResponse(BaseModel):
    typhoon_success: int
    typhoon_errors: int
    typhoon_active_workers: int
    gemini_success: int
    gemini_errors: int
    gemini_active_workers: int
    queue_size: int

# --- Helper Functions for Job Status ---
async def _read_job_statuses() -> Dict[str, Any]:
    async with job_status_lock:
        if not os.path.exists(JOB_STATUS_FILE):
            return {}
        async with aiofiles.open(JOB_STATUS_FILE, mode='r') as f:
            content = await f.read()
            if not content: 
                return {}
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

# --- เพิ่มฟังก์ชันกลับเข้ามา ---
async def notify_webhook(result_data: ProcessingResponse):
    if not N8N_WEBHOOK_URL:
        logger.warning("[BG] ไม่ได้ตั้งค่า N8N_WEBHOOK_URL, ข้ามการส่ง notification")
        return
    
    logger.info(f"[BG] กำลังส่ง Webhook notification ไปที่: {N8N_WEBHOOK_URL}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                N8N_WEBHOOK_URL, 
                json=json.loads(result_data.model_dump_json()), 
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"[BG] ส่ง Webhook notification สำเร็จ! Status: {response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"[BG] เกิดข้อผิดพลาดในการเชื่อมต่อเพื่อส่ง Webhook: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"[BG] Webhook URL ตอบกลับด้วยสถานะผิดพลาด: {e.response.status_code} - {e.response.text}")

# --- Other Helper Functions ---
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

async def get_existing_page_numbers(collection_name: str, source_file_name: str) -> Set[int]:
    global qdrant_client
    if qdrant_client is None: 
        return set()
    
    existing_pages = set()
    try:
        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source", 
                    match=models.MatchValue(value=source_file_name)
                )
            ]
        )
        
        next_offset = None
        while True:
            response, next_offset = await asyncio.to_thread(
                qdrant_client.scroll,
                collection_name=collection_name,
                scroll_filter=source_filter,
                limit=250,
                offset=next_offset,
                with_payload=["metadata.loc.pageNumber"],
                with_vectors=False
            )
            
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

# --- Library Document Info Helpers (ใหม่) ---
async def fetch_document_info(doc_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """
    ดึงข้อมูลเอกสารจาก Library API: /api/documents/{doc_id}
    คืนค่า dict หรือ None ถ้าผิดพลาด
    """
    if not LIBRARY_API_TOKEN:
        logger.warning("LIBRARY_API_TOKEN ไม่ถูกตั้งค่า ไม่สามารถเรียกดูข้อมูลเอกสารได้")
        return None

    info_path = f"api/documents/{doc_id}"
    url = urljoin(LIBRARY_API_BASE_URL, info_path)
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
            return None
    except httpx.HTTPError as e:
        logger.warning(f"เรียกดูข้อมูลเอกสาร {doc_id} ล้มเหลว: {e}")
        return None

def make_source_label_from_doc_info(doc_info: Dict[str, Any]) -> Optional[str]:
    """
    สร้าง label สำหรับ metadata.source = 'title - first_author'
    ถ้าไม่มี authors ให้ใช้ title อย่างเดียว
    """
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

# --- ปรับปรุงฟังก์ชันหลักสำหรับ OCR ---
async def process_and_upsert_single_page(
    doc: fitz.Document, 
    page_num: int, 
    file_name: str, 
    collection_name: str,
    book_id: Optional[str] = None
) -> Tuple[int, int]:
    """ประมวลผลหน้าเดียว ใช้ Dual OCR Manager"""
    global qdrant_client, ocr_manager
    
    page_index = page_num - 1
    logger.info(f"--- [BG] เริ่มประมวลผลหน้า {page_num} ---")
    
    try:
        page = doc.load_page(page_index)
        page_text = await asyncio.to_thread(page.get_text)
        
        # ตรวจสอบว่าต้อง OCR หรือไม่
        if len(page_text.strip()) < 50 and len(page.get_text("words")) < 10:
            logger.info(f"[BG] กำลังใช้ Dual OCR Manager สำหรับหน้า {page_num}...")
            
            # สร้างรูปภาพสำหรับ OCR
            pix = await asyncio.to_thread(page.get_pixmap, dpi=OCR_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # สร้าง OCR Task
            task_id = f"{file_name}_page_{page_num}_{uuid.uuid4().hex[:8]}"
            ocr_task = OCRTask(
                task_id=task_id,
                page_num=page_num,
                image=img,
                file_name=file_name,
                collection_name=collection_name,
                preferred_provider=OCRProvider.TYPHOON  # เริ่มด้วย Typhoon ก่อน
            )
            
            # ส่ง task และรอผลลัพธ์ของ task นั้นๆ โดยตรง
            result = await ocr_manager.submit_ocr_task(ocr_task)
            if result.success:
                page_text = result.text
                logger.info(f"[BG] OCR สำเร็จสำหรับหน้า {page_num} ด้วย {result.provider_used.value}")
            else:
                logger.warning(f"[BG] OCR ล้มเหลวสำหรับหน้า {page_num}: {result.error}")
                page_text = ""
        
        if not page_text.strip():
            logger.info(f"--- [BG] หน้า {page_num} ไม่มีข้อความให้ประมวลผล ---")
            return 0, 0
        
        # แบ่งเป็น chunks
        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        page_tasks = []
        
        for chunk_index_on_page, chunk in enumerate(page_chunks):
            chunk_metadata = {
                "source": file_name,  # ตรงนี้รับเป็น source_label แล้วจาก caller
                "loc": {
                    "pageNumber": page_num,
                    "chunkIndex": chunk_index_on_page
                },
                "chunk_id": str(uuid.uuid4())
            }
            if book_id:
                chunk_metadata["book_id"] = book_id

            task = asyncio.create_task(async_process_text_chunk(chunk, chunk_metadata))
            page_tasks.append(task)
        
        if not page_tasks:
            logger.info(f"--- [BG] หน้า {page_num} ไม่ได้สร้าง chunks ---")
            return 0, 0
        
        # รอผลลัพธ์จากการประมวลผล chunks
        results = await asyncio.gather(*page_tasks, return_exceptions=True)
        successful_points = [r for r in results if r is not None and not isinstance(r, Exception)]
        failed_count = len(page_tasks) - len(successful_points)
        
        # บันทึกลง Qdrant
        if successful_points:
            await asyncio.to_thread(
                qdrant_client.upsert,
                collection_name=collection_name,
                wait=True,
                points=successful_points
            )
            logger.info(f"[BG] บันทึก {len(successful_points)} chunks จากหน้า {page_num} ลง Qdrant สำเร็จ")
        
        logger.info(f"--- [BG] จบการประมวลผลหน้า {page_num} ---")
        return len(successful_points), failed_count
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดรุนแรงขณะประมวลผลหน้า {page_num}: {e}", exc_info=True)
        return 0, 1

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
                with_payload=["metadata.source"],
                with_vectors=False
            )
            
            for point in response:
                source = point.payload.get("metadata", {}).get("source")
                if source:
                    unique_sources.add(source)
            
            if next_offset is None:
                break
        
        return sorted(list(unique_sources))
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดขณะ list sources จาก collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=404, 
            detail=f"Collection '{collection_name}' not found or an error occurred."
        )

async def search_library_for_book(query: str) -> Optional[List[Dict[str, Any]]]:
    search_url = f"{LIBRARY_API_BASE_URL}/search"
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}
    params = {"q": query}
    
    logger.info(f"กำลังค้นหาหนังสือใน Library ด้วยคำว่า: '{query}'...")
    
    try:
        response = await asyncio.to_thread(
            requests.get, search_url, headers=headers, params=params, timeout=30
        )
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
    """ดาวน์โหลดไฟล์จาก Library API แบบใช้ id (/api/documents/{id}/download)"""
    file_id = file_path  # ใช้ชื่อ parameter เดิมเพื่อความเข้ากันได้
    download_path = f"api/documents/{file_id}/download"
    download_url = urljoin(LIBRARY_API_BASE_URL, download_path)
    headers = {"Authorization": f"Bearer {LIBRARY_API_TOKEN}"}
    logger.info(f"กำลังเริ่มดาวน์โหลดไฟล์ id '{file_id}' จาก URL: {download_url}")

    def _blocking_download_with_progress():
        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                # พยายามดึงชื่อไฟล์จาก Content-Disposition ถ้าไม่มีใช้ id
                content_disp = r.headers.get('Content-Disposition', '')
                filename = file_id
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[-1].strip('"')

                total_size_in_bytes = int(r.headers.get('content-length', 0))
                file_in_memory = io.BytesIO()

                if total_size_in_bytes == 0:
                    logger.warning("ไม่พบข้อมูล Content-Length, ไม่สามารถแสดงความคืบหน้าเป็น % ได้")
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
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
                return filename, file_in_memory.getvalue()
        except requests.exceptions.RequestException as e:
            logger.error(f"เกิดข้อผิดพลาดระหว่างดาวน์โหลดไฟล์ id '{file_id}' จาก Library: {e}")
            return None

    result = await asyncio.to_thread(_blocking_download_with_progress)
    if result is not None:
        return result
    else:
        return None

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text: 
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        
        if end >= len(text): 
            break
            
        start += (chunk_size - chunk_overlap)
        start = max(0, start)
    
    return chunks

# --- New: aiohttp-based Embedding ---
async def async_generate_embedding(input_data: List[str]) -> Optional[List[List[float]]]:
    """เรียก Ollama Embedding API แบบ async ด้วย aiohttp"""
    if not input_data:
        return None
    global embedding_http_session
    if embedding_http_session is None:
        # Lazy fallback (ปกติจะถูกสร้างใน lifespan)
        embedding_http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=CONCURRENCY)
        )
    url = OLLAMA_EMBEDDING_URL
    model = OLLAMA_EMBEDDING_MODEL_NAME
    timeout_seconds = 2048
    max_retries = 3

    async def _embed_one(text_to_embed: str) -> Optional[List[float]]:
        for attempt in range(max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                payload = {"model": model, "prompt": text_to_embed}
                async with embedding_http_session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        emb = data.get("embedding")
                        if isinstance(emb, list):
                            return emb
                        else:
                            logger.error("รูปแบบข้อมูล embedding ไม่ถูกต้อง: %s", str(data)[:500])
                            return None
                    elif resp.status in (502, 503, 504):
                        wait_time = 5 * attempt if attempt > 0 else 1
                        logger.warning(f"Embedding API {resp.status}, retry ใน {wait_time} วินาที (ครั้งที่ {attempt}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        txt = (await resp.text())[:500]
                        logger.error(f"Embedding API status {resp.status}: {txt}")
                        return None
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                wait_time = 5 * attempt if attempt > 0 else 1
                if attempt < max_retries:
                    logger.warning(f"ปัญหาการเชื่อมต่อ Embedding API: {e}. retry ใน {wait_time} วินาที (ครั้งที่ {attempt}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"เชื่อมต่อ Embedding API ไม่สำเร็จหลัง retry {max_retries} ครั้ง: {e}")
                    return None
            except Exception as e:
                logger.error(f"ข้อผิดพลาดไม่คาดคิดจาก Embedding API: {e}", exc_info=True)
                return None
        return None

    tasks = [_embed_one(text) for text in input_data]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    valid_embeddings = [e for e in results if e is not None]
    return valid_embeddings if valid_embeddings else None

async def async_process_text_chunk(
    chunk_text: str, 
    chunk_metadata: Dict[str, Any]
) -> models.PointStruct | None:
    global semaphore_embedding_call
    
    if semaphore_embedding_call is None: 
        return None
    
    async with semaphore_embedding_call:
        chunk_id = chunk_metadata.get('chunk_id')
        if not chunk_id:
            logger.error("ไม่พบ chunk_id (UUID) ใน metadata")
            return None
        
        embedding_results = await async_generate_embedding([chunk_text])
        if not (embedding_results and embedding_results[0]):
            logger.warning(f"สร้าง embedding สำหรับ chunk '{chunk_id}' ล้มเหลว")
            return None
        
        payload = {
            "pageContent": chunk_text,
            "metadata": chunk_metadata
        }
        
        return models.PointStruct(
            id=chunk_id,
            vector=embedding_results[0],
            payload=payload
        )

async def ensure_qdrant_collection(collection_name: str):
    global qdrant_client
    if qdrant_client is None: 
        raise RuntimeError("Qdrant client not initialized")
    
    try:
        await asyncio.to_thread(
            qdrant_client.get_collection, 
            collection_name=collection_name
        )
    except Exception:
        # ถ้าไม่มี ให้สร้างขึ้นมาใหม่
        logger.info(f"ไม่พบ Collection '{collection_name}' กำลังสร้าง...")
        
        await asyncio.to_thread(
            qdrant_client.create_collection,
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=QDRANT_VECTOR_SIZE, 
                distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Collection '{collection_name}' สร้างสำเร็จแล้ว.")
        
        # สร้าง Index สำหรับ source
        logger.info(f"กำลังสร้าง Payload Index (Keyword) สำหรับ 'metadata.source'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        # Index สำหรับ pageNumber
        logger.info(f"กำลังสร้าง Payload Index (Integer) สำหรับ 'metadata.loc.pageNumber'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.loc.pageNumber",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        
        # Index สำหรับ chunkIndex
        logger.info(f"กำลังสร้าง Payload Index (Integer) สำหรับ 'metadata.loc.chunkIndex'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.loc.chunkIndex",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        
        # Index สำหรับ pageContent (Full-text search)
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

        # Index สำหรับ book_id (เพื่อให้ filter ได้สะดวก)
        logger.info(f"กำลังสร้าง Payload Index (Keyword) สำหรับ 'metadata.book_id'...")
        await asyncio.to_thread(
            qdrant_client.create_payload_index,
            collection_name=collection_name,
            field_name="metadata.book_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        logger.info(f"สร้าง Payload Index ทั้งหมดสำหรับ collection '{collection_name}' สำเร็จแล้ว.")

async def page_worker_with_semaphore(
    doc: fitz.Document, 
    page_num: int, 
    file_name: str, 
    collection_name: str,
    book_id: Optional[str] = None
) -> Tuple[int, int]:
    global page_processing_semaphore
    
    if page_processing_semaphore is None:
        raise RuntimeError("Page processing semaphore is not initialized.")
    
    async with page_processing_semaphore:
        return await process_and_upsert_single_page(doc, page_num, file_name, collection_name, book_id)

async def process_pdf_in_background(
    file_path_key: str,
    file_bytes: bytes,
    file_name: str,
    pages_str: Optional[str] = None,
    custom_collection_name: Optional[str] = None
):
    """ประมวลผล PDF ในเบื้องหลัง (ใช้ Dual OCR Manager)"""
    original_file_name = file_name.strip()
    cleaned_source_name = os.path.splitext(original_file_name)[0]

    # กำหนด source_label + book_id จาก Library API ถ้าเป็นเคสมาจาก Library
    source_label = cleaned_source_name
    book_id: Optional[str] = None
    try:
        doc_info = await fetch_document_info(file_path_key)
        label = make_source_label_from_doc_info(doc_info) if doc_info else None
        if label:
            source_label = label
            logger.info(f"[BG] ใช้ source_label จาก Library: '{source_label}' (เดิม: '{cleaned_source_name}')")
        else:
            logger.info(f"[BG] ใช้ source_label จากชื่อไฟล์: '{source_label}'")

        if doc_info and "id" in doc_info:
            book_id = str(doc_info["id"])
        elif str(file_path_key).isdigit():
            book_id = str(file_path_key)

    except Exception as e:
        logger.warning(f"[BG] ไม่สามารถดึงข้อมูลเอกสารเพื่อสร้าง source_label/book_id ได้: {e}. ใช้ชื่อไฟล์และไม่ใส่ book_id")

    if custom_collection_name:
        collection_name = custom_collection_name.strip().lower()
    else:
        collection_name = "".join(
            c for c in cleaned_source_name 
            if c.isalnum() or c in ['-', '_']
        ).lower()
        if not collection_name:
            collection_name = f"pdf_doc_{uuid.uuid4().hex}"
    
    await update_job_status(
        file_path_key, 
        "processing", 
        {
            "file_path": original_file_name, 
            "collection_name": collection_name
        }
    )
    
    final_response = None
    doc = None
    
    try:
        await ensure_qdrant_collection(collection_name)
        
        requested_pages = parse_page_string(pages_str)
        # ใช้ source_label ในการตรวจสอบหน้าที่เคยมีแล้ว
        existing_pages = await get_existing_page_numbers(collection_name, source_label)
        
        if existing_pages:
            logger.info(f"[BG] พบหน้าที่ประมวลผลแล้วสำหรับ source นี้: {sorted(list(existing_pages))}")
        
        doc = await asyncio.to_thread(fitz.open, stream=file_bytes, filetype="pdf")
        total_pages_in_doc = len(doc)
        
        if requested_pages:
            pages_to_process = requested_pages - existing_pages
        else:
            all_doc_pages = set(range(1, total_pages_in_doc + 1))
            pages_to_process = all_doc_pages - existing_pages
            
        if not pages_to_process:
            logger.warning(
                f"[BG] หน้าที่ร้องขอทั้งหมดสำหรับไฟล์ '{cleaned_source_name}' "
                f"มีอยู่แล้วใน collection. ข้ามการประมวลผล"
            )
            final_response = ProcessingResponse(
                collection_name=collection_name,
                status="skipped",
                processed_chunks=0,
                failed_chunks=0,
                message="หน้าที่ร้องขอทั้งหมดมีอยู่แล้วใน collection",
                file_name=original_file_name
            )
        else:
            pages_to_iterate = sorted([
                p for p in pages_to_process 
                if 1 <= p <= total_pages_in_doc
            ])
            
            logger.info(
                f"[BG] จะทำการประมวลผลหน้าใหม่ {len(pages_to_iterate)} หน้า "
                f"(สูงสุด {PAGE_CONCURRENCY} หน้าพร้อมกัน): {pages_to_iterate}"
            )
            
            # สร้าง tasks สำหรับประมวลผลหน้า (ใช้ source_label และ book_id)
            page_processing_tasks = []
            for page_num in pages_to_iterate:
                task = asyncio.create_task(
                    page_worker_with_semaphore(doc, page_num, source_label, collection_name, book_id)
                )
                page_processing_tasks.append(task)
            
            # รอผลลัพธ์
            results = await asyncio.gather(*page_processing_tasks)
            
            doc.close()
            doc = None
            
            # คำนวณผลลัพธ์
            total_successful_chunks = sum(res[0] for res in results)
            total_failed_chunks = sum(res[1] for res in results)
            
            if total_successful_chunks > 0:
                message = (
                    f"ประมวลผลและเพิ่ม {total_successful_chunks} chunks ใหม่ "
                    f"ลงใน Qdrant collection '{collection_name}' สำเร็จแล้ว"
                )
                status_code = "success"
            else:
                message = f"ไม่มี chunks ใหม่ใดๆ ถูกประมวลผลสำเร็จสำหรับ {original_file_name}."
                status_code = "warning" if total_failed_chunks == 0 else "error"
            
            final_response = ProcessingResponse(
                collection_name=collection_name,
                status=status_code,
                processed_chunks=total_successful_chunks,
                failed_chunks=total_failed_chunks,
                message=message,
                file_name=original_file_name
            )
        
        await update_job_status(
            file_path_key, 
            "completed", 
            {"result": json.loads(final_response.model_dump_json())}
        )
        
    except Exception as e:
        logger.error(f"[BG Task] เกิดข้อผิดพลาดรุนแรงที่ไม่สามารถจัดการได้: {e}", exc_info=True)
        error_message = f"เกิดข้อผิดพลาดรุนแรงระหว่างการประมวลผล: {e}"
        
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
        if doc is not None:
            doc.close()
            logger.info(f"[BG] ปิดเอกสาร '{original_file_name}' เรียบร้อยแล้ว")
        
        if final_response:
            await notify_webhook(final_response)

# --- FastAPI Endpoints ---
@app.post(
    "/process_pdf/", 
    response_model=AcknowledgementResponse, 
    status_code=status.HTTP_202_ACCEPTED
)
async def process_pdf_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pages: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None)
):
    file_path_key = file.filename
    logger.info(f"ได้รับไฟล์: {file_path_key}, กำลังเพิ่ม Task เข้าสู่เบื้องหลัง...")
    
    file_bytes = await file.read()
    
    await update_job_status(file_path_key, "queued", {"file_path": file_path_key})
    
    background_tasks.add_task(
        process_pdf_in_background,
        file_path_key=file_path_key,
        file_bytes=file_bytes,
        file_name=file.filename,
        pages_str=pages,
        custom_collection_name=collection_name
    )
    
    return AcknowledgementResponse(
        message="Task accepted. Use the file_path to check status.",
        file_path=file_path_key,
        task_status="queued"
    )

@app.get("/collections/{collection_name}/sources", response_model=SourceListResponse)
async def get_sources_in_collection(
    collection_name: str = Path(..., description="ชื่อของ Collection ที่ต้องการตรวจสอบ")
):
    logger.info(f"ได้รับคำขอเพื่อ list sources ใน collection: '{collection_name}'")
    sources = await list_unique_sources_in_collection(collection_name)
    
    return SourceListResponse(
        collection_name=collection_name,
        source_count=len(sources),
        sources=sources
    )

@app.post(
    "/process_from_library/", 
    response_model=AcknowledgementResponse, 
    status_code=status.HTTP_202_ACCEPTED
)
async def process_from_library(
    request: LibrarySearchRequest, 
    background_tasks: BackgroundTasks
):
    logger.info(f"ได้รับคำขอจาก Library: '{request.query}', กำลังดาวน์โหลด...")
    
    found_files = await search_library_for_book(request.query)
    if not found_files:
        raise HTTPException(
            status_code=404, 
            detail=f"ไม่พบหนังสือใน Library: '{request.query}'"
        )
    
    file_to_process = found_files[0]
    file_id = file_to_process.get('id') or file_to_process.get('path')
    if not file_id:
        raise HTTPException(
            status_code=500, 
            detail="ข้อมูลจาก Library API ไม่มี 'id' หรือ 'path' ของไฟล์"
        )
    download_result = await download_book_from_library(file_id)
    if download_result is None:
        raise HTTPException(
            status_code=500, 
            detail=f"ดาวน์โหลดไฟล์ id '{file_id}' จาก Library ล้มเหลว"
        )
    file_name, file_bytes = download_result
    await update_job_status(file_id, "queued", {"file_path": file_name})
    background_tasks.add_task(
        process_pdf_in_background,
        file_path_key=file_id,
        file_bytes=file_bytes,
        file_name=file_name,
        pages_str=request.pages,
        custom_collection_name=request.collection_name
    )
    return AcknowledgementResponse(
        message="Task accepted. Use the file_path to check status.",
        file_path=file_id,
        task_status="queued"
    )

@app.post(
    "/process_by_id/", 
    response_model=AcknowledgementResponse, 
    status_code=status.HTTP_202_ACCEPTED
)
async def process_by_file_path(
    request: ProcessByPathRequest, 
    background_tasks: BackgroundTasks
):
    file_id = request.file_path
    logger.info(f"ได้รับคำขอจาก Path/ID: '{file_id}', กำลังเพิ่ม Task...")
    download_result = await download_book_from_library(file_id)
    if download_result is None:
        raise HTTPException(
            status_code=500, 
            detail=f"ดาวน์โหลดไฟล์ id '{file_id}' จาก Library ล้มเหลว"
        )
    file_name, file_bytes = download_result
    await update_job_status(file_id, "queued", {"file_path": file_name})
    background_tasks.add_task(
        process_pdf_in_background,
        file_path_key=file_id,
        file_bytes=file_bytes,
        file_name=file_name,
        pages_str=request.pages,
        custom_collection_name=request.collection_name
    )
    return AcknowledgementResponse(
        message="Task accepted. Use the file_path to check status.",
        file_path=file_id,
        task_status="queued"
    )

@app.get("/status", response_model=JobStatusResponse)
async def get_job_status(
    file_path: str = Query(..., description="File Path ที่ได้รับจากการส่งไฟล์ (ต้อง URL Encoded)")
):
    statuses = await _read_job_statuses()
    job_details = statuses.get(file_path)
    
    if not job_details:
        raise HTTPException(
            status_code=404, 
            detail=f"Status for file_path '{file_path}' not found."
        )
    
    return JobStatusResponse(file_path=file_path, details=job_details)

@app.get("/ocr_stats", response_model=OCRStatsResponse)
async def get_ocr_statistics():
    """ดูสถิติการใช้งาน OCR APIs"""
    global ocr_manager
    
    if ocr_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="OCR Manager is not available."
        )
    
    stats = ocr_manager.get_stats()
    
    return OCRStatsResponse(
        typhoon_success=stats["typhoon"]["success"],
        typhoon_errors=stats["typhoon"]["error"],
        typhoon_active_workers=stats["typhoon"]["active_workers"],
        gemini_success=stats["gemini"]["success"],
        gemini_errors=stats["gemini"]["error"],
        gemini_active_workers=stats["gemini"]["active_workers"],
        queue_size=stats["queue_size"]
    )

@app.get("/")
async def root():
    return {
        "message": "บริการประมวลผล PDF (Dual OCR) กำลังทำงาน. ใช้ /docs สำหรับเอกสาร API."
    }

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """ตรวจสอบสุขภาพของระบบทั้งหมด"""
    global ocr_manager, qdrant_client
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }
    
    # ตรวจสอบ OCR Manager
    if ocr_manager and ocr_manager.running:
        stats = ocr_manager.get_stats()
        health_status["components"]["ocr_manager"] = {
            "status": "healthy",
            "typhoon_workers": stats["typhoon"]["active_workers"],
            "gemini_workers": stats["gemini"]["active_workers"],
            "queue_size": stats["queue_size"]
        }
    else:
        health_status["components"]["ocr_manager"] = {
            "status": "unhealthy",
            "reason": "OCR Manager not running"
        }
        health_status["status"] = "degraded"
    
    # ตรวจสอบ Qdrant
    try:
        if qdrant_client:
            await asyncio.to_thread(qdrant_client.get_collections)
            health_status["components"]["qdrant"] = {"status": "healthy"}
        else:
            health_status["components"]["qdrant"] = {
                "status": "unhealthy", 
                "reason": "Qdrant client not initialized"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["qdrant"] = {
            "status": "unhealthy", 
            "reason": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # ตรวจสอบ Environment Variables
    missing_vars = []
    required_vars = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "TYPHOON_API_KEY": TYPHOON_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "LIBRARY_API_TOKEN": LIBRARY_API_TOKEN
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        health_status["components"]["environment"] = {
            "status": "unhealthy",
            "missing_variables": missing_vars
        }
        health_status["status"] = "unhealthy"
    else:
        health_status["components"]["environment"] = {"status": "healthy"}
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

# --- OCR Queue Management Endpoints ---
@app.post("/ocr_queue/clear")
async def clear_ocr_queue():
    """ล้างคิว OCR (สำหรับ debugging)"""
    global ocr_manager
    
    if ocr_manager is None:
        raise HTTPException(status_code=503, detail="OCR Manager is not available.")
    
    # ล้างคิวทั้งสองฝั่ง
    cleared_typhoon = 0
    cleared_gemini = 0
    try:
        while True:
            ocr_manager.typhoon_queue.get_nowait()
            cleared_typhoon += 1
    except asyncio.QueueEmpty:
        pass
    
    try:
        while True:
            ocr_manager.gemini_queue.get_nowait()
            cleared_gemini += 1
    except asyncio.QueueEmpty:
        pass
    
    return {
        "message": f"Cleared {cleared_typhoon} Typhoon tasks and {cleared_gemini} Gemini tasks from OCR queues",
        "cleared_tasks": cleared_typhoon + cleared_gemini
    }

@app.get("/ocr_queue/status")
async def get_ocr_queue_status():
    """ดูสถานะคิว OCR แบบละเอียด"""
    global ocr_manager
    
    if ocr_manager is None:
        raise HTTPException(status_code=503, detail="OCR Manager is not available.")
    
    stats = ocr_manager.get_stats()
    
    return {
        "queue_size": stats["queue_size"],
        "typhoon": {
            "total_success": stats["typhoon"]["success"],
            "total_errors": stats["typhoon"]["error"],
            "active_workers": stats["typhoon"]["active_workers"],
            "total_workers": TYPHOON_WORKERS,
            "success_rate": (
                stats["typhoon"]["success"] / 
                (stats["typhoon"]["success"] + stats["typhoon"]["error"])
                if (stats["typhoon"]["success"] + stats["typhoon"]["error"]) > 0 
                else 0
            )
        },
        "gemini": {
            "total_success": stats["gemini"]["success"],
            "total_errors": stats["gemini"]["error"],
            "active_workers": stats["gemini"]["active_workers"],
            "total_workers": GEMINI_WORKERS,
            "success_rate": (
                stats["gemini"]["success"] / 
                (stats["gemini"]["success"] + stats["gemini"]["error"])
                if (stats["gemini"]["success"] + stats["gemini"]["error"]) > 0 
                else 0
            )
        },
        "overall": {
            "total_processed": (
                stats["typhoon"]["success"] + stats["typhoon"]["error"] + 
                stats["gemini"]["success"] + stats["gemini"]["error"]
            ),
            "total_success": stats["typhoon"]["success"] + stats["gemini"]["success"],
            "total_errors": stats["typhoon"]["error"] + stats["gemini"]["error"]
        }
    }

# --- Worker Management Endpoints ---
@app.post("/workers/restart")
async def restart_workers():
    """รีสตาร์ท OCR Workers (สำหรับ maintenance)"""
    global ocr_manager
    
    if ocr_manager is None:
        raise HTTPException(status_code=503, detail="OCR Manager is not available.")
    
    logger.info("กำลังรีสตาร์ท OCR Workers...")
    
    # หยุด workers เดิม
    await ocr_manager.stop_workers()
    
    # รอสักครู่
    await asyncio.sleep(2)
    
    # เริ่ม workers ใหม่
    await ocr_manager.start_workers()
    
    logger.info("รีสตาร์ท OCR Workers สำเร็จ")
    
    return {
        "message": "OCR Workers restarted successfully",
        "typhoon_workers": TYPHOON_WORKERS,
        "gemini_workers": GEMINI_WORKERS
    }

# --- Configuration Endpoints ---
@app.get("/config")
async def get_current_config():
    """ดูการตั้งค่าปัจจุบัน"""
    return {
        "workers": {
            "typhoon_workers": TYPHOON_WORKERS,
            "gemini_workers": GEMINI_WORKERS,
            "page_concurrency": PAGE_CONCURRENCY,
            "embedding_concurrency": CONCURRENCY
        },
        "processing": {
            "ocr_dpi": OCR_DPI,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "storage": {
            "qdrant_vector_size": QDRANT_VECTOR_SIZE
        },
        "apis": {
            "ollama_embedding_url": OLLAMA_EMBEDDING_URL,
            "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL_NAME,
            "library_api_base_url": LIBRARY_API_BASE_URL
        }
    }

# --- Main Execution ---
if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("!!!! โปรดใช้ Python 3.8+ !!!!")
        sys.exit(1)
    
    required_env_vars = [
        GEMINI_API_KEY, 
        TYPHOON_API_KEY, 
        QDRANT_URL, 
        QDRANT_API_KEY, 
        LIBRARY_API_TOKEN
    ]
    
    if not all(required_env_vars):
        logger.critical("ไม่พบตัวแปรสภาพแวดล้อมที่จำเป็น. โปรดตรวจสอบ key.env")
        sys.exit(1)
    
    import uvicorn
    uvicorn.run("pdf_mix:app", host="0.0.0.0", port=8080)