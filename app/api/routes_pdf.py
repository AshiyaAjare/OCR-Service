import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from app.config import settings
from app.models.schemas import (
    ExtractionResult,
    FullAnalysisResponse,
    LLMAnalysisResult,
)
from app.services.extraction_pipeline import extract_pdf_dual
from app.services.ollama_client import call_ollama_mistral

router = APIRouter(prefix="/api/v1/pdf", tags=["pdf"])


def _ensure_upload_dir() -> None:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


async def _save_upload_to_disk(upload: UploadFile) -> str:
    _ensure_upload_dir()
    ext = os.path.splitext(upload.filename or "")[1].lower() or ".pdf"
    temp_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, temp_name)

    contents = await upload.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return file_path


@router.post(
    "/extract-basic",
    response_model=ExtractionResult,
    summary="Dual extraction (PDF text + OCR) without LLM",
)
async def extract_basic(file: UploadFile = File(...)):
    """
    Upload a PDF and get:
    - Per-page text via PDF parsing
    - Per-page OCR text from page snapshots
    - Merged text

    This does NOT call Mistral.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = await _save_upload_to_disk(file)

    try:
        result = await extract_pdf_dual(file_path)
        return result
    finally:
        # Optional: clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post(
    "/extract-with-llm",
    response_model=FullAnalysisResponse,
    summary="Dual extraction + Mistral analysis via Ollama",
)
async def extract_with_llm(
    file: UploadFile = File(...),
    instruction: str = Form(
        "Summarize the key points and return a short JSON with fields "
        "company_name, period, key_financials, dividends, notes."
    ),
):
    """
    Upload a PDF, run dual extraction, and then send the merged text to
    Ollama Mistral with your instruction.

    - 'instruction' is what you want from the LLM (e.g., JSON schema).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = await _save_upload_to_disk(file)

    try:
        extraction = await extract_pdf_dual(file_path)

        llm_prompt = (
            f"{instruction}\n\n"
            "Here is the text extracted from the PDF (both direct parsing and OCR):\n\n"
            f"{extraction.merged_text}"
        )

        llm_response = await call_ollama_mistral(llm_prompt)

        llm_analysis = LLMAnalysisResult(
            instruction=instruction,
            raw_response=llm_response,
        )

        return FullAnalysisResponse(
            extraction=extraction,
            llm_analysis=llm_analysis,
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
