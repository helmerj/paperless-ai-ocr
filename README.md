# Paperless-ngx AI-OCR Bot 🤖

A Python script that extracts documents from Paperless-ngx, processes them using a local LLM (Ollama) for OCR, and updates both the document content text in PaperlessNGX and the stored files.

## Motivation

Paperless-ngx relies on Tesseract for Optical Character Recognition (OCR). However, the OCR quality is often insufficient for complex layouts or handwritten text. Existing tools like `paperless-AI` struggle to generate accurate titles, correspondents, and tags because the underlying OCR text provided as input is poor.

While I managed to get `paperless-GPT` running and working with Ollama, it consistently failed to automatically process documents with the `auto-OCR` tag, and updating the actual text content within Paperless-ngx was only successful during manual processing. This bot solves those gaps by providing a reliable, automated pipeline for high-quality OCR.

## ✨ Features

* **Pagination:** Seamlessly processes thousands of documents without interruption.
* **Local LLM-OCR:** Leverages Ollama (e.g., `minicpm-v`) for high-precision text recognition.
* **Real-time Dashboard:** Console-based progress tracking with percentage completion.
* **Intelligent Caching:** Prevents redundant downloads, saving bandwidth and compute resources.
* **Dead Letter Queue (DLQ): Automatically logs failed document IDs to failed_ids.txt for later analysis.
* **Smart Reprocessing: Use the --retry-failed flag to process only previously failed documents.
* **Fully Automated Workflow:** Automatically tags processed documents with `ocr-done`.
* **External Prompting:** Easily customize AI instructions via the `prompt.md` file.
* **Single Document Targeting:** Process specific documents using the `-id` parameter.
* **Batch Tag Processing:** Target specific subgroups of documents using the `-tag_id` parameter.
* **Force Override:** Use the `--force` flag to re-process documents even if they already carry the `ocr-done` tag.

## 🛠 Installation

### 1. System Dependencies

Ensure `poppler` is installed (required by `pdf2image`):

* **macOS:** `brew install poppler`
* **Ubuntu/Debian:** `sudo apt-get install poppler-utils`

### 2. Python Environment

```bash
# Clone the repository
git clone https://github.com/helmerj/paperless-ai-ocr.git
cd paperless-ai-ocr

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Configuration

**Copy the example environment file to `.env`:**
`cp env.example .env`

**Configure your Paperless-ngx URL, API Token, Ollama URL, and Model:**
`nano .env`

**Ensure the specified LLM model is downloaded:**
`ollama pull minicpm-v:latest`

**Ensure to specify the number of cores to use to greatly speed up OCR processing in the .env file
`NUMBER_CORES=4`

**Note:** Verify that the `TAG_ID` in your `.env` matches the numerical ID for your "ocr-done" tag in Paperless-ngx.

### 4. 🚀 Usage

**Process all documents that DO NOT have the `ocr-done` tag:**
`python run.py`

**Process a specific document by ID:**
`python run.py -id 1234`

**Force re-process a specific document (even if already tagged):**
`python run.py -id 1234 --force`

**Process all documents containing a specific tag (e.g., Tag ID 123):**
`python run.py -tag_id 123`

**Force re-process all documents in a tag group (ignoring the `ocr-done` tag):**
`python run.py -tag_id 123 --force`

**Reprocess failed documents (DLQ):
`python run.py --retry-failed`
<br>
This reads from failed_ids.txt, attempts processing, and clears the file upon start.

## 5. 📝 License

MIT