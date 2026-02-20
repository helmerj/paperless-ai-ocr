# Paperless-ngx AI-OCR Bot 🤖

Python-Skript, dass Dokumente aus Paperless-ngx extrahiert, mittels lokalem LLM (Ollama) verarbeitet und die Texte sowie die Dateien aktualisiert.

## ✨ Features
- **Pagination:** Verarbeitet tausende Dokumente ohne Unterbrechung.
- **Lokales LLM-OCR:** Nutzt Ollama (z.B. `minicpm-v`) für hochpräzise Texterkennung.
- **Echtzeit-Dashboard:** Fortschrittsanzeige in Prozent direkt in der Konsole.
- **Intelligentes Caching:** Verhindert redundante Downloads und schont damit Ressourcen.
- **Vollautomatischer Workflow:** Markiert Dokumente nach Abschluss mit einem Tag (`ocr-done`).
- **Externer Prompt:** Anweisungen an die KI können einfach über `prompt.md` angepasst werden.

## 🛠 Installation

### 1. System-Abhängigkeiten
Stelle sicher, dass `poppler` installiert ist (wird von `pdf2image` benötigt):
- **macOS:** `brew install poppler`
- **Ubuntu/Debian:** `sudo apt-get install poppler-utils`

### 2. Python Umgebung
```bash
# Repository klonen
git clone [https://github.com/dein-username/paperless-ai-ocr.git](https://github.com/helmerj/paperless-ai-ocr.git)
cd paperless-ai-ocr

# Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

### 3. Konfiguration
# Kopiere die Datei env.example nach .env.
cp env.example .env

# Trage deine Paperless-URL, Paperless API-Token, Ollama-URL und Ollama Modell ein.
nano .env

# Stelle sicher, dass das gelistet LLM model installiert ist

ollama pull minicpm-v:latest

# Stelle sicher, dass die TAG_ID deiner ID für "ocr-done" entspricht.

### 4. 🚀 Start
python run.py

### 5. 📝 Lizenz
MIT

