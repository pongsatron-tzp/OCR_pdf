from dotenv import load_dotenv
load_dotenv()

import os
import io
import re
import json
import base64
import tempfile
from typing import Optional, List, Dict, Tuple
from PIL import Image

# Marker (core)
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser


# ---------- Utilities ----------
def _safe_filename(name: str) -> str:
    keep = "-_.() []"
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in keep).strip()
    return cleaned or "document"


def _marker_build_config(
    output_format: str,
    page_range: Optional[str],
    llm_model: Optional[str] = None
) -> Dict[str, object]:
    """
    สร้าง config สำหรับ Marker (วิธีที่ 2 นี้ ไม่ได้ใช้ LLM ของ Marker)
    - output_format: "markdown" หรือ "json"
    - page_range: string ของช่วงหน้า (0-based) เช่น "0-4,9"
    - ocr_languages: ตัวเลือกภาษา OCR (เช่น "tha+eng") หากเวอร์ชัน Marker รองรับ
    """
    cfg = {"output_format": output_format}
    if page_range:
        cfg["page_range"] = page_range
    cfg["force_ocr"] = True
    return cfg

# โหลด artifacts ครั้งเดียว (cache)
MARKER_ARTIFACTS = create_model_dict()


def marker_convert(file_bytes: bytes, output_format: str, page_range: Optional[str] = None, llm_model: Optional[str] = None):
    """
    เรียก Marker เพื่อแปลง PDF -> format ที่ต้องการ
    คืนค่า (text_or_json_str, ext, images_dict)
    """
    cfg = _marker_build_config(output_format=output_format, page_range=page_range, llm_model=llm_model)
    config_parser = ConfigParser(cfg)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        converter = PdfConverter(
            artifact_dict=MARKER_ARTIFACTS,
            config=config_parser.generate_config_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(tmp_path)
        return text_from_rendered(rendered)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------- Image helpers ----------
def _walk_images_from_blocks(block: Dict[str, object], collector: List[Tuple[str, str]]):
    imgs = block.get("images") or {}
    if isinstance(imgs, dict):
        for blk_id, b64 in imgs.items():
            if isinstance(blk_id, str) and isinstance(b64, str):
                collector.append((blk_id, b64))
    for ch in (block.get("children") or []):
        if isinstance(ch, dict):
            _walk_images_from_blocks(ch, collector)


def extract_images_from_json_blocks(blocks: List[Dict[str, object]]) -> List[Tuple[str, str, int]]:
    results: List[Tuple[str, str, int]] = []
    if not isinstance(blocks, list):
        return results
    for page_idx, page_block in enumerate(blocks):
        if not isinstance(page_block, dict):
            continue
        tmp: List[Tuple[str, str]] = []
        _walk_images_from_blocks(page_block, tmp)
        for blk_id, b64 in tmp:
            results.append((blk_id, b64, page_idx))
    return results


def _decode_base64_image(b64: str) -> bytes:
    s = (b64 or "").strip()
    if s.startswith("data:"):
        try:
            s = s.split(",", 1)[1]
        except Exception:
            pass
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return base64.b64decode(s)


def save_pil_images_dict(
    images_dict: Dict[str, Image.Image],
    output_root: str,
    source_label: str
) -> List[Dict[str, object]]:
    saved: List[Dict[str, object]] = []
    page_dir = os.path.join(output_root, source_label, "page_unknown")
    os.makedirs(page_dir, exist_ok=True)
    for idx, (fname, pil) in enumerate(images_dict.items()):
        try:
            fpath = os.path.join(page_dir, f"img_{idx:03d}.png")
            pil.save(fpath, format="PNG")
            saved.append({"path": fpath, "page": None, "block_id": fname})
        except Exception:
            continue
    return saved


def save_images_from_blocks(
    blocks: List[Dict[str, object]],
    output_root: str,
    source_label: str
) -> List[Dict[str, object]]:
    saved: List[Dict[str, object]] = []
    images_info = extract_images_from_json_blocks(blocks)
    for idx, (blk_id, b64, page_idx0) in enumerate(images_info):
        page_num = page_idx0 + 1
        page_dir = os.path.join(output_root, source_label, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)
        try:
            raw = _decode_base64_image(b64)
            with Image.open(io.BytesIO(raw)) as pil:
                fpath = os.path.join(page_dir, f"img_{idx:03d}.png")
                pil.save(fpath, format="PNG")
            saved.append({"path": fpath, "page": page_num, "block_id": blk_id})
        except Exception:
            continue
    return saved


# ---------- Public API (แปลง PDF -> Markdown + รูป) ----------
def extract_markdown_and_images(
    file_bytes: bytes,
    page_range: Optional[str] = None,
    image_output_dir: Optional[str] = None,
    source_label: Optional[str] = None,
    google_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Tuple[str, List[Dict[str, object]]]:
    """
    คืนค่า (markdown_str, images_saved)
    โดย images_saved รวมจากทั้ง images_dict และ JSON blocks
    """
    old_key = os.environ.get("GOOGLE_API_KEY")
    set_temp_key = bool(google_api_key and google_api_key.strip())
    if set_temp_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key.strip()
    try:
        # 1) Markdown (พร้อม images_dict)
        md_text, _, images_dict = marker_convert(file_bytes, output_format="markdown", page_range=page_range, llm_model=None)
        markdown_str = md_text if isinstance(md_text, str) else ""
        # 2) JSON blocks
        blocks_json, _, _ = marker_convert(file_bytes, output_format="json", page_range=page_range, llm_model=None)
        try:
            blocks = json.loads(blocks_json) if isinstance(blocks_json, str) else blocks_json
        except Exception:
            blocks = []
        images_saved: List[Dict[str, object]] = []
        if image_output_dir:
            src_label = _safe_filename(source_label or "document")
            if isinstance(images_dict, dict) and images_dict:
                images_saved.extend(save_pil_images_dict(images_dict, output_root=image_output_dir, source_label=src_label))
            images_saved.extend(save_images_from_blocks(blocks, output_root=image_output_dir, source_label=src_label))
        return markdown_str, images_saved
    finally:
        if set_temp_key:
            if old_key is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = old_key


def save_markdown(markdown_str: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_str)


# ---------- (TH+EN): โพสต์โปรเซส Markdown ด้วย Gemini ----------
def _split_markdown_preserving_codeblocks(text: str) -> List[Tuple[bool, str]]:
    """
    แยก Markdown ออกเป็นชิ้นๆ โดยรักษา code blocks (``` ... ```) ไว้ไม่ให้แตะต้อง
    คืนค่าเป็นลิสต์ของทูเพิล (is_code_block, segment)
    """
    pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    parts: List[Tuple[bool, str]] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append((False, text[last:m.start()]))
        parts.append((True, text[m.start():m.end()]))
        last = m.end()
    if last < len(text):
        parts.append((False, text[last:]))
    return parts


def _chunk_text(s: str, max_chars: int) -> List[str]:
    """
    แบ่งข้อความยาวเป็นชิ้น โดยพยายามตัดตามย่อหน้า/บรรทัดก่อน
    """
    if len(s) <= max_chars:
        return [s]

    chunks = []
    buf = []
    curr = 0
    lines = s.splitlines(keepends=True)
    for ln in lines:
        if curr + len(ln) > max_chars and curr > 0:
            chunks.append("".join(buf))
            buf = [ln]
            curr = len(ln)
        else:
            buf.append(ln)
            curr += len(ln)
    if buf:
        chunks.append("".join(buf))
    return chunks


# ป้องกันการแก้ไข inline code (`...`) โดยใช้ placeholder
INLINE_CODE_TOKEN = "§CODE_{:04d}§"
URL_TOKEN = "§URL_{:04d}§"

def _mask_inline_code(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    idx = 0

    def repl(m: re.Match) -> str:
        nonlocal idx
        idx += 1
        token = INLINE_CODE_TOKEN.format(idx)
        mapping[token] = m.group(0)  # เก็บทั้ง backticks
        return token

    # จับ `...` ที่ไม่คร่อมหลายบรรทัด
    masked = re.sub(r"`[^`\n]+`", repl, text)
    return masked, mapping


def _mask_urls(text: str) -> Tuple[str, Dict[str, str]]:
    """
    ป้องกันการแก้ไขลิงก์/URL ทั้งแบบ [text](url) และแบบเปลือย http(s)://...
    """
    mapping: Dict[str, str] = {}
    idx = 0

    # 1) Mask URLs ใน Markdown link: [text](url)
    def repl_md_link(m: re.Match) -> str:
        nonlocal idx
        text_part = m.group(1)
        url_part = m.group(2)
        idx += 1
        token = URL_TOKEN.format(idx)
        mapping[token] = url_part
        return f"[{text_part}]({token})"

    masked = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_md_link, text)

    # 2) Mask bare URLs (ไม่อยู่ในวงเล็บของ markdown link)
    def repl_bare_url(m: re.Match) -> str:
        nonlocal idx
        url = m.group(0)
        if url.startswith("§URL_") and url.endswith("§"):
            return url
        idx += 1
        token = URL_TOKEN.format(idx)
        mapping[token] = url
        return token

    masked = re.sub(r"https?://[^\s)>\]]+", repl_bare_url, masked)
    return masked, mapping


def _unmask(text: str, mapping: Dict[str, str]) -> str:
    for token in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(token, mapping[token])
    return text


def refine_markdown_with_gemini(
    markdown_str: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    max_chunk_chars: int = 24000
) -> str:
    """
    ส่ง Markdown ให้ Gemini ช่วยแก้คำผิด OCR/จัดเว้นวรรค/ตัดคำ ทั้งไทยและอังกฤษ โดยคงโครงสร้าง Markdown
    - ไม่แตะต้องโค้ดบล็อก (```...```)
    - ป้องกันการแก้ไข inline code (`...`) และ URL/ลิงก์ ด้วย placeholder
    - ถ้าข้อความยาว จะตัดเป็นชิ้นแล้วรวมกลับ
    """
    try:
        import google.generativeai as genai
    except Exception:
        return markdown_str

    key = (api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        return markdown_str

    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)

    system_prompt = (
        "You are a bilingual (Thai and English) copy editor. Improve readability and fix OCR artifacts "
        "while strictly preserving original meaning and Markdown structure.\n"
        "- Do NOT summarize, omit, or add content. Keep technical terms, acronyms, units as-is.\n"
        "- Preserve Markdown: headings (#, ##, ###), bullets, ordered lists, tables, links, images.\n"
        "- Do NOT change fenced code blocks (```...```).\n"
        "- Do NOT change inline code or URLs. Special tokens like §CODE_xxxx§ and §URL_xxxx§ must remain unchanged.\n"
        "- Thai: fix word segmentation, spacing, broken lines; maintain Thai punctuation.\n"
        "- English: fix spacing, common OCR typos; keep original dialect/spelling (e.g., colour vs color).\n"
        "- Merge lines broken by OCR when appropriate; keep original paragraphs where possible.\n"
        "- Output Markdown only."
    )

    segments = _split_markdown_preserving_codeblocks(markdown_str)
    out_parts: List[str] = []

    for is_code, seg in segments:
        if is_code or not seg.strip():
            out_parts.append(seg)
            continue

        # 1) ป้องกัน inline code และ URL ไม่ให้ถูกแก้ไข
        seg_masked_1, code_map = _mask_inline_code(seg)
        seg_masked_2, url_map = _mask_urls(seg_masked_1)

        # 2) ตัดข้อความเป็นชิ้นเพื่อเรียกโมเดล
        chunks = _chunk_text(seg_masked_2, max_chunk_chars)
        chunk_outputs: List[str] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            prompt = f"{system_prompt}\n\n---\nSection {i}/{total}:\n{chunk}"
            try:
                resp = model.generate_content(prompt)
                cleaned = getattr(resp, "text", "") or ""
                chunk_outputs.append(cleaned if cleaned.strip() else chunk)
            except Exception:
                chunk_outputs.append(chunk)

        processed = "".join(chunk_outputs)

        # 3) คืนค่า placeholder กลับ
        processed = _unmask(processed, url_map)
        processed = _unmask(processed, code_map)

        out_parts.append(processed)

    return "".join(out_parts)


# ---------- ตัวอย่างการใช้งาน ----------
if __name__ == "__main__":
    from pathlib import Path

    pdf_path = "C:/path/to/your.pdf"
    out_md   = "outputs/sample_refined.md"
    out_img  = "outputs/images"

    file_bytes = Path(pdf_path).read_bytes()

    # 1) แปลง PDF -> Markdown และดึงรูป (OCR ไทย+อังกฤษ หาก Marker รองรับ ocr_languages)
    markdown_str, images = extract_markdown_and_images(
        file_bytes=file_bytes,
        page_range="0-20",
        image_output_dir=out_img,
        source_label="sample",
        google_api_key=None,
        llm_model=None
    )

    # 2) โพสต์โปรเซส Markdown ด้วย Gemini ให้รองรับทั้งไทยและอังกฤษ
    refined_md = refine_markdown_with_gemini(
        markdown_str,
        api_key=None,
        model_name="gemini-2.5-flash",
        max_chunk_chars=24000
    )

    # 3) บันทึกผลลัพธ์
    save_markdown(refined_md, out_md)

    print(f"Saved refined MD to: {out_md}")
    print(f"Saved {len(images)} images under: {out_img}")