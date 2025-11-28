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
```

Python:

```bash
pip install -r requirements.txt
```

## Configuration

The service uses environment variables for configuration. Create a `.env` file in the `pdf_extractor` directory with the following variables:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://192.168.2.23:11435
OLLAMA_MODEL=mistral

# Optional: Upload directory for temporary PDF storage
# UPLOAD_DIR=/tmp/pdf_extractor
```

### Environment Variables

- **OLLAMA_BASE_URL**: The base URL for your Ollama API server (default: `http://192.168.2.23:11435`)
- **OLLAMA_MODEL**: The name of the Ollama model to use for LLM processing (default: `mistral`)
- **UPLOAD_DIR**: Directory for temporarily storing uploaded PDFs (default: `/tmp/pdf_extractor`)

**Note**: The `.env` file is already included in `.gitignore` to keep your configuration private.
