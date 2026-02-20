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
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from tqdm import tqdm
from pdf2image import convert_from_bytes

# --- EMERGENCY BOOT PRINT ---
print("0. Interpreter reached the script start.", flush=True)

load_dotenv()

CONFIG = {
    "URL": os.getenv("PAPERLESS_URL", "").rstrip("/"),
    "TOKEN": os.getenv("PAPERLESS_TOKEN"),
    "OLLAMA_URL": os.getenv("OLLAMA_URL"),
    "MODEL": os.getenv("MODEL", "minicpm-v:latest"),
    "TAG_NAME": os.getenv("TAG_NAME", "ocr-done"),
    "TAG_ID": os.getenv("TAG_ID", 1065),
    "BUFFER_SIZE": int(os.getenv("BUFFER_SIZE", 5)),
    "PAGE_LIMIT": int(os.getenv("PAGE_LIMIT", 3)),
    "CACHE_DIR": "./ocr_cache"
}

class StatusTracker:
    def __init__(self):
        self.total_to_process = 0
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()

    def set_total(self, count):
        with self.lock:
            self.total_to_process = count

    def increment_done(self, success=True):
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1

    def print_report(self):
        with self.lock:
            percent = (self.completed / self.total_to_process * 100) if self.total_to_process > 0 else 0
            print("\n" + "="*40)
            print(f"📊 STATUS REPORT")
            print(f"✅ Completed: {self.completed} / {self.total_to_process} ({percent:.1f}%)")
            if self.failed > 0:
                print(f"❌ Failures:  {self.failed}")
            print("="*40 + "\n", flush=True)

tracker = StatusTracker()

class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_doc_path(self, doc_id):
        return os.path.join(self.cache_dir, str(doc_id))

    def is_cached(self, doc_id):
        return os.path.exists(self.get_doc_path(doc_id))

    def save_to_cache(self, doc_id, images, total_pages):
        path = self.get_doc_path(doc_id)
        os.makedirs(path, exist_ok=True)
        for i, img_bytes in enumerate(images):
            with open(os.path.join(path, f"page_{i}.png"), "wb") as f:
                f.write(img_bytes)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"total_pages": total_pages}, f)

    def load_from_cache(self, doc_id):
        path = self.get_doc_path(doc_id)
        images = []
        page_files = sorted([f for f in os.listdir(path) if f.endswith(".png")])
        for pf in page_files:
            with open(os.path.join(path, pf), "rb") as f:
                images.append(f.read())
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        return images, meta["total_pages"]

    def clear_cache(self, doc_id):
        path = self.get_doc_path(doc_id)
        if os.path.exists(path):
            shutil.rmtree(path)

class PaperlessAPI:
    def __init__(self, config):
        self.base_url = config["URL"]
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {config['TOKEN']}"})

    def fetch_documents(self, exclude_tag=None, include_tag=None, force=False):
        """
        Fetches documents based on tag filters.
        If force=True, we do not exclude documents that already have the 'done' tag.
        """
        params = {"page_size": 100}
        
        # Only exclude the 'done' tag if we aren't forcing re-processing
        if exclude_tag and not force:
            params["tags__id__none"] = exclude_tag
            
        if include_tag:
            params["tags__id__all"] = include_tag

        url = f"{self.base_url}/api/documents/"
        is_first_page = True
        
        while url:
            try:
                r = self.session.get(url, params=params if is_first_page else None, timeout=30)
                r.raise_for_status()
                data = r.json()
                
                if is_first_page:
                    tracker.set_total(data.get("count", 0))
                    print(f"🎯 Total documents found for criteria: {tracker.total_to_process}")
                    is_first_page = False
                
                for doc in data.get("results", []):
                    yield doc
                
                url = data.get("next")
            except Exception as e:
                print(f"❌ API connection failed: {e}", flush=True)
                url = None

    def get_document_metadata(self, doc_id):
        r = self.session.get(f"{self.base_url}/api/documents/{doc_id}/")
        r.raise_for_status()
        return r.json()

    def download_document(self, doc_id):
        return self.session.get(f"{self.base_url}/api/documents/{doc_id}/download/").content

    def update_document(self, doc_id, text, tags):
        self.session.patch(f"{self.base_url}/api/documents/{doc_id}/", 
                           json={"content": text, "tags": tags}).raise_for_status()

    def replace_file(self, doc_id, pdf_bytes):
        if 'csrftoken' not in self.session.cookies:
            self.session.get(f"{self.base_url}/api/")
        headers = {
            "X-CSRFToken": self.session.cookies.get('csrftoken', ''),
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/"
        }
        files = {"document": ("ocr_fixed.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        r = self.session.post(f"{self.base_url}/api/documents/{doc_id}/replace_document/", 
                             headers=headers, files=files, timeout=120)
        return r.status_code == 200

class OllamaClient:
    def __init__(self, config):
        self.url = config["OLLAMA_URL"]
        self.model = config["MODEL"]
        try:
            with open("prompt.md", "r") as f: 
                self.prompt = f.read()
        except: 
            self.prompt = "Transcribe the text in this image."

    def ocr_image(self, img_bytes):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {"model": self.model, "prompt": self.prompt, "images": [b64], "stream": False}
        try:
            r = requests.post(self.url, json=payload, timeout=300)
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"[Error: {e}]"

class PDFProcessor:
    @staticmethod
    def to_images(pdf_bytes, page_limit=3):
        images = convert_from_bytes(pdf_bytes)
        total_pages = len(images)
        limited_selection = images[:page_limit]
        byte_images = []
        for img in limited_selection:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_images.append(buf.getvalue())
        return byte_images, total_pages

    @staticmethod
    def from_text(text_pages):
        writer = PdfWriter()
        for text in text_pages:
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            t = c.beginText(40, 800)
            t.setFont("Helvetica", 10)
            for line in text.split("\n"): 
                t.textLine(line[:100])
            c.drawText(t)
            c.showPage()
            c.save()
            buf.seek(0)
            writer.add_page(PdfReader(buf).pages[0])
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

def producer(api, cache, job_queue, config_tag_id, target_id=None, force=False, subgroup_tag_id=None):
    """
    Discovery logic with precedence and force handling.
    """
    done_tag_id = int(config_tag_id)

    if target_id:
        try:
            doc = api.get_document_metadata(target_id)
            current_tags = [int(t) for t in doc.get('tags', [])]
            
            if done_tag_id in current_tags and not force:
                print(f"⚠️ Document {target_id} already has the done tag. Use --force to override.")
                tracker.set_total(1)
                tracker.increment_done()
                job_queue.put(None)
                return
                
            docs_to_process = [doc]
            tracker.set_total(1)
        except Exception as e:
            print(f"❌ Error fetching ID {target_id}: {e}")
            job_queue.put(None)
            return
    else:
        # fetch_documents handles API-level filtering for subgroup_tag_id and exclude logic
        docs_to_process = api.fetch_documents(
            exclude_tag=done_tag_id, 
            include_tag=subgroup_tag_id,
            force=force
        )

    for doc in docs_to_process:
        doc_id = doc['id']
        current_tags = [int(t) for t in doc.get('tags', [])]
        
        try:
            if cache.is_cached(doc_id):
                imgs, total_count = cache.load_from_cache(doc_id)
            else:
                raw_pdf = api.download_document(doc_id)
                imgs, total_count = PDFProcessor.to_images(raw_pdf, page_limit=CONFIG["PAGE_LIMIT"])
                cache.save_to_cache(doc_id, imgs, total_count)
            
            job_queue.put({
                "id": doc_id, 
                "title": doc['title'], 
                "images": imgs, 
                "total_pages": total_count, 
                "tags": current_tags
            })
        except Exception as e:
            print(f"❌ Error on {doc['title']}: {e}")
            tracker.increment_done(success=False)
    
    job_queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="Paperless-ngx AI OCR Tool")
    parser.add_argument("-id", type=int, help="Process a single document ID")
    parser.add_argument("-tag_id", type=int, help="Process all documents with this Tag ID")
    parser.add_argument("--force", action="store_true", help="Process even if the 'done' tag is present")
    args = parser.parse_args()

    print("🚀 Script initiated...", flush=True)
    api = PaperlessAPI(CONFIG)
    ollama = OllamaClient(CONFIG)
    cache = CacheManager(CONFIG["CACHE_DIR"])
    
    job_queue = queue.Queue(maxsize=CONFIG["BUFFER_SIZE"])
    
    threading.Thread(
        target=producer, 
        args=(api, cache, job_queue, CONFIG["TAG_ID"], args.id, args.force, args.tag_id), 
        daemon=True
    ).start()

    while True:
        job = job_queue.get()
        if job is None: break
        
        print(f"🧠 Processing: {job['title']} (ID: {job['id']})...")
        texts = [ollama.ocr_image(img) for img in tqdm(job['images'], leave=False)]
        
        try:
            footer = f"\n\n--- OCR Footer: {len(job['images'])} pages processed ---"
            final_content = "\n\n".join(texts) + footer
            new_pdf = PDFProcessor.from_text(texts)
            api.replace_file(job['id'], new_pdf)
            api.update_document(job['id'], final_content, list(set(job['tags'] + [CONFIG["TAG_ID"]])))
            cache.clear_cache(job['id'])
            tracker.increment_done(success=True)
        except Exception as e:
            print(f"❌ Update failed for {job['id']}: {e}")
            tracker.increment_done(success=False)
            
        tracker.print_report()
        job_queue.task_done()

if __name__ == "__main__":
    main()