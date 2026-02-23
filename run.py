import io
import os
import shutil
import base64
import queue
import threading
import json
import requests
import sys
import argparse
import logging
import time
import httpx
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pdf2image import convert_from_bytes
from logging.handlers import RotatingFileHandler

# Load Environment Variables
load_dotenv()

# --- Configuration ---
CONFIG = {
    "URL": os.getenv("PAPERLESS_URL", "").rstrip("/"),
    "TOKEN": os.getenv("PAPERLESS_TOKEN"),
    "OLLAMA_URL": os.getenv("OLLAMA_URL"),
    "MODEL": os.getenv("MODEL", "minicpm-v:latest"),
    # Docling Config
    "DOCLING_URL": os.getenv("DOCLING_URL", "http://localhost:5001/v1alpha/convert/source"),
    "DOCLING_LANGS": os.getenv("DOCLING_LANGS", "de,en").split(","),
    "TAG_ID": int(os.getenv("TAG_ID", 1065)),
    "FAILED_TAG_ID": int(os.getenv("FAILED_TAG_ID", 1066)),
    "BUFFER_SIZE": int(os.getenv("BUFFER_SIZE", 5)),
    "PAGE_LIMIT": int(os.getenv("PAGE_LIMIT", 3)),
    "CACHE_DIR": "./ocr_cache",
    "THREADS": int(os.getenv("NUMBER_CORES", 1)),
    "OLLAMA_TIMEOUT": int(os.getenv("OLLAMA_TIMEOUT", 600)),
    "OLLAMA_RETRIES": int(os.getenv("OLLAMA_RETRIES", 2)),
    "DLQ_FILE": "failed_ids.txt",
    "LOG_FILE": "ocr_bot.log",
    "LOG_MAX_BYTES": int(os.getenv("LOG_MAX_MB", 10)) * 1024 * 1024,
    "LOG_BACKUPS": int(os.getenv("LOG_BACKUP_COUNT", 5))
}

# --- Logging Setup ---
logger = logging.getLogger("OCR_Bot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

file_handler = RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=CONFIG["LOG_MAX_BYTES"], backupCount=CONFIG["LOG_BACKUPS"])
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- Status Tracker ---
class StatusTracker:
    def __init__(self):
        self.total_to_process = 0
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()

    def set_total(self, count):
        with self.lock: self.total_to_process = count

    def increment_done(self, success=True):
        with self.lock:
            self.completed += 1
            if not success: self.failed += 1
            percent = (self.completed / self.total_to_process * 100) if self.total_to_process > 0 else 0
            sys.stdout.write(f"\r📊 Progress: {self.completed}/{self.total_to_process} ({percent:.1f}%) | Failures: {self.failed}")
            sys.stdout.flush()

tracker = StatusTracker()
dlq_lock = threading.Lock()

def log_to_dlq(doc_id):
    with dlq_lock:
        with open(CONFIG["DLQ_FILE"], "a") as f:
            f.write(f"{doc_id}\n")

# --- Clients ---

class DoclingClient:
    """Production-ready client for docling-serve v1 with Base64 sanitization"""
    def __init__(self, config):
        self.url = config.get("DOCLING_URL", "http://localhost:5001/v1/convert/file")
        self.ocr_engine = os.getenv("DOCLING_OCR_ENGINE", "easyocr")
        
        langs = config.get("DOCLING_LANGS", "en")
        # Ensure we have a list of strings for the API
        self.langs = langs.split(",") if "," in langs else [langs]

    def _sanitize_markdown(self, text):
        """
        Removes any Base64 encoded images that escaped the server-side 
        placeholder setting.
        """
        if not text:
            return ""
        
        # This regex finds Markdown image syntax containing Base64 data:
        # ![Image](data:image/png;base64,...)
        base64_pattern = r'!\[.*?\]\(data:image\/[a-zA-Z]*;base64,[a-zA-Z0-9\/+=\s]*\)'
        
        sanitized_text = re.sub(base64_pattern, '[IMAGE-PLACEHOLDER]', text)
        
        # Double-check for raw data URIs that might not be in Markdown tags
        raw_uri_pattern = r'data:image\/[a-zA-Z]*;base64,[a-zA-Z0-9\/+=\s]*'
        sanitized_text = re.sub(raw_uri_pattern, '[IMAGE-DATA-STRIPPED]', sanitized_text)
        
        return sanitized_text

    def ocr_image(self, file_bytes):
        payload = {
            "to_formats": ["md"],
            "do_ocr": True,
            "ocr_engine": self.ocr_engine,
            "ocr_lang": self.langs,
            "table_mode": "fast",
            "image_export_mode": "placeholder",
            "include_images": False
        }

        files = {
            "files": ("document.pdf", file_bytes, "application/pdf")
        }
        data = {
            "parameters": json.dumps(payload)
        }

        with httpx.Client(timeout=300.0) as client:
            try:
                r = client.post(self.url, files=files, data=data)
                
                if r.status_code != 200:
                    logger.error(f"Docling Error {r.status_code}: {r.text}")
                    raise Exception(f"Docling conversion failed: {r.text}")

                res_json = r.json()
                
                # Navigate v1 response
                doc = res_json.get("document", {})
                markdown = doc.get("md_content") or doc.get("outputs", {}).get("md", "")
                
                # --- FAIL-SAFE SANITIZATION ---
                # This ensures your Paperless DB stays clean even if the server ignores our 'placeholder' request
                clean_markdown = self._sanitize_markdown(markdown)

                if not clean_markdown:
                    return "Warning: Docling processed the file but returned no text."

                return clean_markdown.strip()

            except Exception as e:
                logger.error(f"Docling Client Failure: {e}")
                raise

class OllamaClient:
    def __init__(self, config):
        self.url = config["OLLAMA_URL"]
        self.model = config["MODEL"]
        self.timeout = config["OLLAMA_TIMEOUT"]
        self.max_retries = config["OLLAMA_RETRIES"]
        try:
            with open("prompt.md", "r") as f: self.prompt = f.read()
        except: self.prompt = "Transcribe the text in this image."

    def ocr_image(self, img_bytes):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {"model": self.model, "prompt": self.prompt, "images": [b64], "stream": False}
        
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    wait = (attempt + 1) * 10
                    logger.warning(f"Ollama Timeout. Retrying in {wait}s... ({attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                else: raise e

# --- Rest of Helper Classes (PaperlessAPI, CacheManager, PDFProcessor) ---
# (Keeping these logic blocks the same as your original script)

class PaperlessAPI:
    def __init__(self, config):
        self.base_url = config["URL"]
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {config['TOKEN']}"})

    def fetch_documents(self, exclude_tag=None, include_tag=None, force=False):
        params = {"page_size": 100}
        if exclude_tag and not force: params["tags__id__none"] = exclude_tag
        if include_tag: params["tags__id__all"] = include_tag
        url = f"{self.base_url}/api/documents/"
        is_first_page = True
        while url:
            r = self.session.get(url, params=params if is_first_page else None, timeout=30)
            r.raise_for_status()
            data = r.json()
            if is_first_page:
                tracker.set_total(data.get("count", 0))
                is_first_page = False
            for doc in data.get("results", []): yield doc
            url = data.get("next")

    def get_document_metadata(self, doc_id):
        return self.session.get(f"{self.base_url}/api/documents/{doc_id}/").json()

    def download_document(self, doc_id):
        return self.session.get(f"{self.base_url}/api/documents/{doc_id}/download/").content

    def update_document(self, doc_id, text=None, tags=None):
        payload = {}
        if text is not None: payload["content"] = text
        if tags is not None: payload["tags"] = tags
        self.session.patch(f"{self.base_url}/api/documents/{doc_id}/", json=payload).raise_for_status()

    def replace_file(self, doc_id, pdf_bytes):
        if 'csrftoken' not in self.session.cookies: self.session.get(f"{self.base_url}/api/")
        headers = {"X-CSRFToken": self.session.cookies.get('csrftoken', ''), "Origin": self.base_url, "Referer": f"{self.base_url}/"}
        files = {"document": ("ocr_fixed.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        r = self.session.post(f"{self.base_url}/api/documents/{doc_id}/replace_document/", headers=headers, files=files, timeout=120)
        return r.status_code == 200

class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)
    def get_doc_path(self, doc_id): return os.path.join(self.cache_dir, str(doc_id))
    def is_cached(self, doc_id): return os.path.exists(self.get_doc_path(doc_id))
    def save_to_cache(self, doc_id, images, total_pages):
        path = self.get_doc_path(doc_id)
        os.makedirs(path, exist_ok=True)
        for i, img_bytes in enumerate(images):
            with open(os.path.join(path, f"page_{i}.png"), "wb") as f: f.write(img_bytes)
        with open(os.path.join(path, "meta.json"), "w") as f: json.dump({"total_pages": total_pages}, f)
    def load_from_cache(self, doc_id):
        path = self.get_doc_path(doc_id)
        images = [open(os.path.join(path, f), "rb").read() for f in sorted(os.listdir(path)) if f.endswith(".png")]
        with open(os.path.join(path, "meta.json"), "r") as f: meta = json.load(f)
        return images, meta["total_pages"]
    def clear_cache(self, doc_id):
        path = self.get_doc_path(doc_id)
        if os.path.exists(path): shutil.rmtree(path)

class PDFProcessor:
    @staticmethod
    def to_images(pdf_bytes, page_limit=3):
        images = convert_from_bytes(pdf_bytes)
        limited = images[:page_limit]
        bytes_list = []
        for img in limited:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            bytes_list.append(buf.getvalue())
        return bytes_list, len(images)
    @staticmethod
    def from_text(text_pages):
        writer = PdfWriter()
        for text in text_pages:
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            t = c.beginText(40, 800)
            t.setFont("Helvetica", 10)
            for line in text.split("\n"): t.textLine(line[:100])
            c.drawText(t)
            c.showPage()
            c.save()
            buf.seek(0)
            writer.add_page(PdfReader(buf).pages[0])
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

# --- Execution Logic ---

def producer(api, cache, job_queue, config_tag_id, target_id=None, force=False, subgroup_tag_id=None, retry_failed=False):
    done_tag_id = int(config_tag_id)
    
    # 1. Gather documents based on the mode (Target ID, Retry, or Bulk)
    if retry_failed:
        if not os.path.exists(CONFIG["DLQ_FILE"]):
            for _ in range(CONFIG["THREADS"]): job_queue.put(None)
            return
        with open(CONFIG["DLQ_FILE"], "r") as f:
            ids = list(set(line.strip() for line in f if line.strip()))
        open(CONFIG["DLQ_FILE"], "w").close()
        tracker.set_total(len(ids))
        docs_to_process = []
        for did in ids:
            try: docs_to_process.append(api.get_document_metadata(did))
            except: logger.error(f"Doc {did} metadata fetch failed.")
    elif target_id:
        try:
            doc = api.get_document_metadata(target_id)
            tracker.set_total(1)
            docs_to_process = [doc]
        except:
            for _ in range(CONFIG["THREADS"]): job_queue.put(None)
            return
    else:
        # fetch_documents handles the initial filter, but we verify again below for safety
        docs_to_process = api.fetch_documents(exclude_tag=done_tag_id, include_tag=subgroup_tag_id, force=force)

    # 2. Filter and Queue logic
    for doc in docs_to_process:
        doc_id = doc['id']
        doc_tags = doc.get('tags', [])
        
        # --- THE GATEKEEPER ---
        # Skip if TAG_ID is present AND force is False
        if done_tag_id in doc_tags and not force:
            logger.info(f"⏭️ Skipping Doc {doc_id}: OCR-Done tag already present.")
            tracker.increment_done(success=True) # Mark as "done" for progress bar
            continue

        try:
            if cache.is_cached(doc_id):
                imgs, total_count = cache.load_from_cache(doc_id)
            else:
                raw_pdf = api.download_document(doc_id)
                imgs, total_count = PDFProcessor.to_images(raw_pdf, page_limit=CONFIG["PAGE_LIMIT"])
                cache.save_to_cache(doc_id, imgs, total_count)
            
            job_queue.put({"id": doc_id, "title": doc['title'], "images": imgs, "tags": doc_tags})
        except Exception as e:
            logger.error(f"Producer error for {doc_id}: {e}")
            log_to_dlq(doc_id)
            tracker.increment_done(success=False)
    
    # Signal workers to exit
    for _ in range(CONFIG["THREADS"]): job_queue.put(None)

def worker(api, ocr_client, cache, job_queue):
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        
        doc_id = job['id']
        try:
            logger.info(f"Processing Doc {doc_id} using {ocr_client.__class__.__name__}...")
            texts = [ocr_client.ocr_image(img) for img in job['images']]
            new_pdf = PDFProcessor.from_text(texts)
            
            api.replace_file(doc_id, new_pdf)
            
            final_tags = list((set(job['tags']) | {CONFIG["TAG_ID"]}) - {CONFIG["FAILED_TAG_ID"]})
            api.update_document(doc_id, "\n\n".join(texts), final_tags)
            
            cache.clear_cache(doc_id)
            logger.info(f"✅ Doc {doc_id} complete.")
            tracker.increment_done(success=True)
        except Exception as e:
            logger.error(f"❌ Doc {doc_id} failed: {e}")
            log_to_dlq(doc_id)
            try:
                fail_tags = list(set(job['tags']) | {CONFIG["TAG_ID"], CONFIG["FAILED_TAG_ID"]})
                api.update_document(doc_id, tags=fail_tags)
            except: pass
            tracker.increment_done(success=False)
        finally:
            job_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Multi-Backend Parallel OCR Bot")
    parser.add_argument("-id", type=int, help="Target ID")
    parser.add_argument("-tag_id", type=int, help="Target Tag Group")
    parser.add_argument("--force", action="store_true", help="Force process")
    parser.add_argument("--retry-failed", action="store_true", help="Retry DLQ")
    parser.add_argument("--docling", action="store_true", help="Use Docling-serve instead of Ollama")
    args = parser.parse_args()

    api = PaperlessAPI(CONFIG)
    cache = CacheManager(CONFIG["CACHE_DIR"])
    job_queue = queue.Queue(maxsize=CONFIG["BUFFER_SIZE"])

    # Strategy Pattern: Select OCR Client
    if args.docling:
        ocr_client = DoclingClient(CONFIG)
        logger.info("🚀 Starting in DOCLING mode")
    else:
        ocr_client = OllamaClient(CONFIG)
        logger.info("🚀 Starting in OLLAMA mode")

    # Producer thread
    threading.Thread(target=producer, args=(api, cache, job_queue, CONFIG["TAG_ID"], args.id, args.force, args.tag_id, args.retry_failed), daemon=True).start()

    # Worker threads
    threads = [threading.Thread(target=worker, args=(api, ocr_client, cache, job_queue)) for _ in range(CONFIG["THREADS"])]
    for t in threads: t.start()
    for t in threads: t.join()
    print("\n🏁 Process finished.")

if __name__ == "__main__":
    main()