import io
import os
import shutil
import base64
import queue
import threading
import json
import requests
import sys
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
    "PAGE_LIMIT": 3,
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

    def fetch_all_documents(self, tag_id):
        url = f"{self.base_url}/api/documents/?tags__id__none={tag_id}&page_size=100"
        is_first_page = True
        
        while url:
            try:
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                
                if is_first_page:
                    tracker.set_total(data.get("count", 0))
                    print(f"🎯 Total documents found to process: {tracker.total_to_process}")
                    is_first_page = False
                
                results = data.get("results", [])
                for doc in results:
                    yield doc
                
                url = data.get("next")
            except Exception as e:
                print(f"❌ API connection failed: {e}", flush=True)
                url = None

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

def producer(api, cache, job_queue, tag_id):
    for doc in api.fetch_all_documents(tag_id):
        doc_id = doc['id']
        current_tags = doc.get('tags', [])
        
        if tag_id in current_tags:
            tracker.increment_done() # Skip but count as "processed"
            continue
            
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
            print(f"❌ Producer error on {doc['title']}: {e}", flush=True)
            tracker.increment_done(success=False)
    
    job_queue.put(None)

def main():
    print("🚀 Main function starting...", flush=True)
    api = PaperlessAPI(CONFIG)
    ollama = OllamaClient(CONFIG)
    cache = CacheManager(CONFIG["CACHE_DIR"])
    tag_id = CONFIG["TAG_ID"]
    
    job_queue = queue.Queue(maxsize=CONFIG["BUFFER_SIZE"])
    threading.Thread(target=producer, args=(api, cache, job_queue, tag_id), daemon=True).start()

    while True:
        job = job_queue.get()
        if job is None: break
        
        doc_id = job['id']
        print(f"🧠 Processing: {job['title']}...")
        
        texts = []
        for img in tqdm(job['images'], leave=False):
            texts.append(ollama.ocr_image(img))
        
        try:
            footer = f"\n\n--- OCR Footer: {len(job['images'])} of {job['total_pages']} pages processed ---"
            final_content = "\n\n".join(texts) + footer
            new_pdf = PDFProcessor.from_text(texts)
            api.replace_file(doc_id, new_pdf)
            api.update_document(doc_id, final_content, list(set(job['tags'] + [tag_id])))
            cache.clear_cache(doc_id)
            tracker.increment_done(success=True)
        except Exception as e:
            print(f"❌ Update failed for {job['title']}: {e}", flush=True)
            tracker.increment_done(success=False)
            
        tracker.print_report()
        job_queue.task_done()
    
    print("🏁 All documents processed. Final Status:", flush=True)
    tracker.print_report()

if __name__ == "__main__":
    main()