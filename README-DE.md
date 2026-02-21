# Paperless-ngx AI-OCR Bot 🤖

Python-Skript, dass Dokumente aus Paperless-ngx extrahiert, mittels lokalem LLM (Ollama) verarbeitet und die Texte sowie die Dateien aktualisiert.

## Motivation
PaperlessNGX benutzt Tesseract zur Testerkennung (OCR). Ich war mit der OCR-Qualität nicht wirklich zufrieden.  Tools wie paperless-AI waren nur bedingt in der Lage wirklich gute Titel, Korrespondenten und Tags zu generieren weil der gespeicherte Text als input nicht gut genug war. 
Ich habe paperless-GPT mit Ollama zwar zum Laufen gebracht, es hat sich allerdings beständig geweigert Dokumente mit auto-OCR Tag automatisch zu prozessieren und eine Aktualisierung des Textinhalts in PaperlessNGX war nur bei manuellem Prozessieren erfolgreich.

## ✨ Features
- **Pagination:** Verarbeitet tausende Dokumente ohne Unterbrechung.
- **Lokales LLM-OCR:** Nutzt Ollama (z.B. `minicpm-v`) für hochpräzise Texterkennung.
- **Echtzeit-Dashboard:** Fortschrittsanzeige in Prozent direkt in der Konsole.
- **Intelligentes Caching:** Verhindert redundante Downloads und schont damit Ressourcen.
- **Vollautomatischer Workflow:** Markiert Dokumente nach Abschluss mit einem Tag (`ocr-done`).
- **Dead Letter Queue (DLQ):** Fehlgeschlagene Dokument-IDs werden automatisch in failed_ids.txt gespeichert.
- **OCR-Failed Tag in PaperlessNGX:** Verfolge fehlgeschlagene OCR Verusche in PaperlessNGX
- **Intelligente Wiederholung:** Mit dem --retry-failed Flag können gezielt nur die Fehlversuche erneut prozessiert werden.
- **Externer Prompt:** Anweisungen an die KI können einfach über `prompt.md` angepasst werden.
- **Auswahl per Dokument-ID:** Einzelene Dokumente können mit dem -id Paramter zur Prozessierung ausgewählt werden: run.py -id XXX
- **Auswahl per Tag-ID** Gruppen von Dukuemnten können per Tag-ID ausgewählt werden: run.py -tag_id XXX
- **Force-Parameter** : Umgehung des 'ocr-done' Tag Checks: run.py -id xxx --force oder run.py -tag_id XXX --force
- **Logging:** Logging zur Konsole und in eine Log-Datei

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
```

### 3. Konfiguration
**Kopiere die Datei env.example nach .env.**
<br>
`cp env.example .env`

**Trage deine Paperless-URL, Paperless API-Token, Ollama-URL und Ollama Modell ein.**
<br>
`nano .env`

**Stelle sicher, dass das gelistet LLM model installiert ist**
<br>
`ollama pull minicpm-v:latest`

**Stelle sicher, dass die Anzahl der Prozessorkerne in der .env Datei definiert ist um die OCR Prozessierung zu beschleunigen.**
`NUMBER_CORES=4`

**Stelle sicher, dass die TAG_ID deiner ID für "ocr-done" entspricht.**



### 4. 🚀 Start
**Prozessiere alle Dokumente die NICHT den 'ocr-done' Tag haben**
<br>
`python run.py`

**Prozessiere das Dokument mit der id 1234**
<br>
`python run.py -id 1234`

**Prozessiere das Dokument mit der id 1234 egal ob es das Tag 'ocr-done' Tag hat oder nicht**
<br>
`python run.py -id 1234 --force`

**Prozessiere alle Dokumente mit dem Tag <tag name> (tag id von <tag name> in paperless == 123)**
<br>
`python run.py -tag_id 123`

**Prozessiere alle Dokumente mit dem Tag <tag name> (tag id von <tag name> in paperless == 123) egal ob sie auch das 'ocr-done' Tag haben**
<br>
`python run.py -tag_id 123 --force`

**Fehlgeschlagene Dokumente erneut versuchen (DLQ):**
<br>
`python run.py --retry-failed`
Liest die IDs aus failed_ids.txt, startet die Verarbeitung und leert die Datei. Bei einem erfolgreichen erneuten Versuch (via --retry-failed) wird das ocr-failed Tag automatisch entfernt und durch das ocr-done Tag ersetzt.

## 5. 📝 Lizenz
MIT

