import pytest
import requests_mock
import queue
import os
from run import PaperlessAPI, StatusTracker, CacheManager, OllamaClient, PDFProcessor, producer

# --- StatusTracker Tests ---
def test_status_tracker():
    st = StatusTracker()
    st.set_total(5)
    # Korrektur: Die Methode heißt increment_done
    st.increment_done(success=True)
    st.increment_done(success=False)
    assert st.completed == 2
    assert st.failed == 1

# --- CacheManager Tests ---
def test_cache_manager(tmp_path):
    cm = CacheManager(str(tmp_path))
    doc_id = 99
    cm.save_to_cache(doc_id, [b"fake_data"], 1)
    assert cm.is_cached(doc_id) is True
    
    imgs, count = cm.load_from_cache(doc_id)
    assert count == 1
    cm.clear_cache(doc_id)
    assert cm.is_cached(doc_id) is False

# --- PaperlessAPI Tests ---
def test_api_fetch_all(requests_mock):
    config = {"URL": "http://test", "TOKEN": "key"}
    api = PaperlessAPI(config)
    # Korrektur: Die URL und Methode an run.py anpassen
    requests_mock.get("http://test/api/documents/?tags__id__none=1065&page_size=100", 
                      json={"count": 1, "next": None, "results": [{"id": 1, "title": "Doc", "tags": []}]})
    
    docs = list(api.fetch_all_documents(1065))
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc"

# --- OllamaClient Tests ---
def test_ollama_client_error(requests_mock):
    config = {"OLLAMA_URL": "http://ollama", "MODEL": "test"}
    client = OllamaClient(config)
    requests_mock.post("http://ollama", exc=Exception("Ollama Down"))
    
    result = client.ocr_image(b"fake")
    assert "[Error:" in result

# --- PDFProcessor Tests (Boostet die Coverage) ---
def test_pdf_processor_from_text():
    texts = ["Seite 1 Inhalt", "Seite 2 Inhalt"]
    pdf_bytes = PDFProcessor.from_text(texts)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0

def test_producer_with_faulty_pdf(requests_mock, tmp_path):
    """Prüft, ob der Producer bei einem PDF-Fehler nicht abbricht."""
    config = {
        "URL": "http://test", 
        "TOKEN": "key", 
        "PAGE_LIMIT": 3, 
        "CACHE_DIR": str(tmp_path)
    }
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    # 1. Mocke ein Dokument
    requests_mock.get("http://test/api/documents/?tags__id__none=1065&page_size=100", 
                      json={"count": 1, "next": None, "results": [{"id": 99, "title": "Kaputtes PDF", "tags": []}]})
    
    # 2. Mocke den Download
    requests_mock.get("http://test/api/documents/99/download/", content=b"kein_echtes_pdf")
    
    # 3. Simuliere einen Fehler im PDFProcessor
    with pytest.MonkeyPatch().context() as m:
        def mock_to_images(pdf_bytes, page_limit):
            raise ValueError("PDF ist korrupt!")
        
        m.setattr("run.PDFProcessor.to_images", mock_to_images)
        
        # Führe den Producer aus (innerhalb des Tests direkt)
        docs = list(api.fetch_all_documents(1065))
        
        # Der Producer sollte den Fehler loggen und weitermachen (oder increment_done(False) rufen)
        # Wir prüfen hier, ob die Queue leer bleibt oder der Prozess nicht crasht
        producer(api, cache, job_queue, 1065)
        
        # Der Producer setzt am Ende ein 'None' in die Queue (Signal zum Beenden)
        first_item = job_queue.get()
        assert first_item is None 
        # Da das Dokument fehlerhaft war, darf es NICHT als Job in der Queue landen    