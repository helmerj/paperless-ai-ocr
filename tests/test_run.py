import pytest
import requests_mock
import queue
import os
from run import PaperlessAPI, StatusTracker, CacheManager, OllamaClient, PDFProcessor, producer

# --- StatusTracker Tests ---
def test_status_tracker():
    st = StatusTracker()
    st.set_total(5)
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
    # Changed to match the URL parameters and method name in run.py
    requests_mock.get("http://test/api/documents/?page_size=100&tags__id__none=1065", 
                      json={"count": 1, "next": None, "results": [{"id": 1, "title": "Doc", "tags": []}]})
    
    docs = list(api.fetch_documents(exclude_tag=1065))
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc"

# --- OllamaClient Tests ---
def test_ollama_client_error(requests_mock):
    config = {"OLLAMA_URL": "http://ollama", "MODEL": "test"}
    client = OllamaClient(config)
    requests_mock.post("http://ollama", exc=Exception("Ollama Down"))
    
    result = client.ocr_image(b"fake")
    assert "[Error:" in result

# --- PDFProcessor Tests ---
def test_pdf_processor_from_text():
    texts = ["Seite 1 Inhalt", "Seite 2 Inhalt"]
    pdf_bytes = PDFProcessor.from_text(texts)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0

def setup_producer_mocks(requests_mock, doc_id, tags, title="Test Doc"):
    """Helper to mock metadata and download for producer tests"""
    requests_mock.get(f"http://test/api/documents/{doc_id}/", 
                      json={"id": doc_id, "title": title, "tags": tags})
    requests_mock.get(f"http://test/api/documents/{doc_id}/download/", 
                      content=b"%PDF-1.4 fake content")

def test_producer_with_faulty_pdf(requests_mock, tmp_path):
    config = {
        "URL": "http://test", 
        "TOKEN": "key", 
        "PAGE_LIMIT": 3, 
        "CACHE_DIR": str(tmp_path)
    }
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    requests_mock.get("http://test/api/documents/?page_size=100&tags__id__none=1065", 
                      json={"count": 1, "next": None, "results": [{"id": 99, "title": "Kaputtes PDF", "tags": []}]})
    requests_mock.get("http://test/api/documents/99/download/", content=b"kein_echtes_pdf")
    
    with pytest.MonkeyPatch().context() as m:
        def mock_to_images(pdf_bytes, page_limit=3):
            raise ValueError("PDF ist korrupt!")
        
        m.setattr("run.PDFProcessor.to_images", mock_to_images)
        producer(api, cache, job_queue, config_tag_id=1065)
        
        # Expecting only the None sentinel because the doc failed
        first_item = job_queue.get()
        assert first_item is None 

# --- New Test Cases for -id and -tag_id ---

def test_producer_single_id_skips_if_tagged(requests_mock, tmp_path):
    config = {"URL": "http://test", "TOKEN": "key", "PAGE_LIMIT": 1, "CACHE_DIR": str(tmp_path)}
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    setup_producer_mocks(requests_mock, 123, [1065])
    producer(api, cache, job_queue, config_tag_id=1065, target_id=123, force=False)
    
    assert job_queue.get() is None

def test_producer_single_id_force_processes(requests_mock, tmp_path):
    config = {"URL": "http://test", "TOKEN": "key", "PAGE_LIMIT": 1, "CACHE_DIR": str(tmp_path)}
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    setup_producer_mocks(requests_mock, 123, [1065])
    
    with pytest.MonkeyPatch().context() as m:
        # Fixed lambda to accept keyword arguments
        m.setattr("run.PDFProcessor.to_images", lambda pdf, page_limit=1: ([b"img"], 1))
        producer(api, cache, job_queue, config_tag_id=1065, target_id=123, force=True)
    
    job = job_queue.get()
    assert job is not None
    assert job['id'] == 123

def test_producer_tag_id_subgroup_filtering(requests_mock, tmp_path):
    config = {"URL": "http://test", "TOKEN": "key", "PAGE_LIMIT": 1, "CACHE_DIR": str(tmp_path)}
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    expected_url = "http://test/api/documents/?page_size=100&tags__id__none=1065&tags__id__all=50"
    requests_mock.get(expected_url, json={"count": 1, "next": None, "results": [{"id": 456, "title": "Subgroup Doc", "tags": [50]}]})
    requests_mock.get("http://test/api/documents/456/download/", content=b"pdf")

    with pytest.MonkeyPatch().context() as m:
        m.setattr("run.PDFProcessor.to_images", lambda pdf, page_limit=1: ([b"img"], 1))
        producer(api, cache, job_queue, config_tag_id=1065, subgroup_tag_id=50, force=False)
    
    job = job_queue.get()
    assert job is not None
    assert job['id'] == 456

def test_producer_tag_id_with_force(requests_mock, tmp_path):
    config = {"URL": "http://test", "TOKEN": "key", "PAGE_LIMIT": 1, "CACHE_DIR": str(tmp_path)}
    api = PaperlessAPI(config)
    cache = CacheManager(str(tmp_path))
    job_queue = queue.Queue()
    
    # Force=True removes tags__id__none=1065 from params
    requests_mock.get("http://test/api/documents/?page_size=100&tags__id__all=50", 
                      json={"count": 1, "next": None, "results": [{"id": 789, "title": "Forced Doc", "tags": [50, 1065]}]})
    requests_mock.get("http://test/api/documents/789/download/", content=b"pdf")

    with pytest.MonkeyPatch().context() as m:
        m.setattr("run.PDFProcessor.to_images", lambda pdf, page_limit=1: ([b"img"], 1))
        producer(api, cache, job_queue, config_tag_id=1065, subgroup_tag_id=50, force=True)
    
    job = job_queue.get()
    assert job is not None
    assert job['id'] == 789