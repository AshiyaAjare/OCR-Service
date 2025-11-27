# PDF Dual Extraction Service

FastAPI service that runs **two parallel extractions** on PDFs:

1. **Text-based extraction** using a PDF parser
2. **Image-based extraction** using OCR on page snapshots

Then optionally sends the merged text to **Ollama Mistral** running locally.

## Requirements

System:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
