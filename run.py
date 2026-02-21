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
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pdf2image import convert_from_bytes

# Load Environment Variables
load_dotenv()

# --- Configuration ---
CONFIG = {
    "URL": os.getenv("PAPERLESS_URL", "").rstrip("/"),
    "TOKEN": os.getenv("PAPERLESS_TOKEN"),
    "OLLAMA_URL": os.getenv("OLLAMA_URL"),
    "MODEL": os.getenv("MODEL", "minicpm-v:latest"),
    "TAG_ID": int(os.getenv("TAG_ID", 1065)),
    "FAILED_TAG_ID": int(os.getenv("FAILED_TAG_ID", 1066)),
    "BUFFER_SIZE": int(os.getenv("BUFFER_SIZE", 5)),
    "PAGE_LIMIT": int(os.getenv("PAGE_LIMIT", 3)),
    "CACHE_DIR": "./ocr_cache",
    "THREADS": int(os.getenv("NUMBER_CORES", 1)),
    "OLLAMA_TIMEOUT": int(os.getenv("OLLAMA_TIMEOUT", 600)),
    "OLLAMA_RETRIES": int(os.getenv("OLLAMA_RETRIES", 2)),
    "DLQ_FILE": "failed_ids.txt",
    "LOG_FILE": "ocr_bot.log"
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Thread Safety
dlq_lock = threading.Lock()

def log_to_dlq(doc_id):
    """Appends a failed ID to the dead letter queue file safely."""
    with dlq_lock:
        with open(CONFIG["DLQ_FILE"], "a") as f:
            f.write(f"{doc_id}\n")

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

# --- Core Modules ---

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
        b64 = base64.get_encoder().encode(img_bytes).decode("utf-8") if hasattr(base64, 'get_encoder') else base64.b64encode(img_bytes).decode("utf-8")
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
        docs_to_process = api.fetch_documents(exclude_tag=done_tag_id, include_tag=subgroup_tag_id, force=force)

    for doc in docs_to_process:
        try:
            doc_id = doc['id']
            if cache.is_cached(doc_id):
                imgs, total_count = cache.load_from_cache(doc_id)
            else:
                raw_pdf = api.download_document(doc_id)
                imgs, total_count = PDFProcessor.to_images(raw_pdf, page_limit=CONFIG["PAGE_LIMIT"])
                cache.save_to_cache(doc_id, imgs, total_count)
            job_queue.put({"id": doc_id, "title": doc['title'], "images": imgs, "tags": doc.get('tags', [])})
        except Exception as e:
            logger.error(f"Producer error for {doc.get('id')}: {e}")
            log_to_dlq(doc.get('id'))
            tracker.increment_done(success=False)
    
    for _ in range(CONFIG["THREADS"]): job_queue.put(None)

def worker(api, ollama, cache, job_queue):
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        
        doc_id = job['id']
        try:
            logger.info(f"Processing Doc {doc_id}...")
            texts = [ollama.ocr_image(img) for img in job['images']]
            new_pdf = PDFProcessor.from_text(texts)
            
            api.replace_file(doc_id, new_pdf)
            
            # SUCCESS: Done present, remove Failed
            final_tags = list((set(job['tags']) | {CONFIG["TAG_ID"]}) - {CONFIG["FAILED_TAG_ID"]})
            api.update_document(doc_id, "\n\n".join(texts), final_tags)
            
            cache.clear_cache(doc_id)
            logger.info(f"✅ Doc {doc_id} complete.")
            tracker.increment_done(success=True)
        except Exception as e:
            logger.error(f"❌ Doc {doc_id} failed: {e}")
            log_to_dlq(doc_id)
            # FAILURE: Done present, add Failed
            try:
                fail_tags = list(set(job['tags']) | {CONFIG["TAG_ID"], CONFIG["FAILED_TAG_ID"]})
                api.update_document(doc_id, tags=fail_tags)
            except: pass
            tracker.increment_done(success=False)
        finally:
            job_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="M4 Mac Mini Parallel OCR Bot")
    parser.add_argument("-id", type=int, help="Target ID")
    parser.add_argument("-tag_id", type=int, help="Target Tag Group")
    parser.add_argument("--force", action="store_true", help="Force process")
    parser.add_argument("--retry-failed", action="store_true", help="Retry DLQ")
    args = parser.parse_args()

    api = PaperlessAPI(CONFIG)
    ollama = OllamaClient(CONFIG)
    cache = CacheManager(CONFIG["CACHE_DIR"])
    job_queue = queue.Queue(maxsize=CONFIG["BUFFER_SIZE"])

    # Producer thread
    threading.Thread(target=producer, args=(api, cache, job_queue, CONFIG["TAG_ID"], args.id, args.force, args.tag_id, args.retry_failed), daemon=True).start()

    # Worker threads
    threads = [threading.Thread(target=worker, args=(api, ollama, cache, job_queue)) for _ in range(CONFIG["THREADS"])]
    for t in threads: t.start()
    for t in threads: t.join()
    print("\n🏁 Process finished.")

if __name__ == "__main__":
    main()