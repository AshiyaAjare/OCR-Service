import os
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.schemas import (
    ExtractionResult,
    FullAnalysisResponse,
    LLMAnalysisResult,
)
from app.models.db_models import ExtractedDocumentModel
from app.services.extraction_pipeline import extract_pdf_dual, truncate_text_for_llm, process_pdf_from_file
from app.services.ollama_client import call_ollama_mistral, extract_json_from_text
from app.services.fallback_extractors import apply_fallback_extractors

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


@router.get(
    "/extracted-document/{document_id}",
    summary="Get status and basic info for an extracted document",
)
def get_extracted_document_status(
    document_id: int,
    db: Session = Depends(get_db),
):
    doc = (
        db.query(ExtractedDocumentModel)
        .filter(ExtractedDocumentModel.id == document_id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    response = {
        "id": doc.id,
        "source_email_id": doc.source_email_id,
        "source_url": doc.source_url,
        "file_path": doc.file_path,
        "file_type": doc.file_type,
        "num_pages": doc.num_pages,
        "is_processed": doc.is_processed,
        "processing_status": doc.processing_status,
        #top-level metadata fields
        "ticker": doc.ticker,
        "company_name": doc.company_name,
        "report_date": doc.report_date.isoformat() if doc.report_date else None,
        "period": doc.period,
        "document_type": doc.document_type,
        "source_domain": doc.source_domain,
        "language": doc.language,
        "ocr_confidence": doc.ocr_confidence,
        "ingestion_method": doc.ingestion_method,
    }

    if doc.is_processed and doc.processing_status == "completed":
        response["extraction result"]={
            "merged_text": doc.merged_text,
            "extraction_metadata": doc.extraction_metadata
        }
    return response


@router.post(
    "/extract-with-llm",
    summary="Dual extraction + Mistral analysis via Ollama (saves to database)",
)
async def extract_with_llm(
    file: UploadFile = File(...),
    instruction: str = Form(
        default=(
            "You are analyzing a financial or regulatory PDF. "
            "Return a single JSON object with these top-level keys:\n"
            "- ticker, company_name, report_date (YYYY-MM-DD), period, document_type,\n"
            "- source_domain, language, ocr_confidence (0..1), ingestion_method,\n"
            "- broker_estimate: number or null (broker's equity target price per share if mentioned),\n"
            "- broker_estimate_detail: object or null with fields:\n"
            "  * value: number (same as broker_estimate - the equity target price per share)\n"
            "  * estimate_type: string (e.g., 'target_price')\n"
            "  * currency: string (e.g., 'INR')\n"
            "  * as_of_date: string in 'YYYY-MM-DD' format or null\n"
            "  * broker_name: string (e.g., 'ICICI Securities') or null\n"
            "  * rating: string (e.g., 'HOLD', 'BUY', 'SELL') or null\n"
            "  * horizon: string (e.g., '12M (I-Sec rating framework)') or null\n"
            "  Example: If document contains 'Target Price: INR 1,110 ... HOLD', return:\n"
            "  {\"broker_estimate\": 1110.0, \"broker_estimate_detail\": {\"value\": 1110.0, \"estimate_type\": \"target_price\", \"currency\": \"INR\", \"rating\": \"HOLD\"}}\n"
            "- tables, kv_pairs, headings, detected_tickers, raw_dates_found.\n"
            "Only output valid JSON. Do not include explanations or comments."
        ),
        description=(
            "Instruction passed to the LLM. Override if you need a custom schema."
        ),
    ),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF, run dual extraction, send to Ollama Mistral for analysis,
    and save the results to the ExtractedDocument table.
    
    This endpoint saves data to the database similar to the email-resolve API.

    - 'instruction' is what you want from the LLM (e.g., JSON schema).
    
    Returns response in the same format as email-resolve API:
    {
        "url": filename or None,
        "resolved_type": "pdf",
        "action": "extraction_completed" | "extraction_failed",
        "detail": {
            "document_id": int,
            "extraction": ExtractionResult,
            "llm_analysis": LLMAnalysisResult
        }
    }
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = await _save_upload_to_disk(file)
    result = {
        "url": file.filename,
        "resolved_type": "pdf",
        "action": None,
        "detail": {}
    }

    try:
        # Use process_pdf_from_file to save to database (similar to email-resolve API)
        # Pass the instruction in meta so it uses the custom instruction
        meta = {
            "llm_instruction": instruction,
            "ingestion_method": "pdf",
        }
        
        # process_pdf_from_file is synchronous, so we need to run it in an executor
        loop = asyncio.get_event_loop()
        document_id = await loop.run_in_executor(
            None, 
            process_pdf_from_file, 
            file_path, 
            meta
        )
        
        # Fetch the saved document to get extraction results for response
        doc = (
            db.query(ExtractedDocumentModel)
            .filter(ExtractedDocumentModel.id == document_id)
            .first()
        )
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found after processing")
        
        # Build extraction result from saved document
        extraction_metadata = doc.extraction_metadata or {}
        llm_structured = extraction_metadata.get("scalar_metadata", {})
        
        # Create ExtractionResult-like structure from saved data
        # Note: We don't have the full page-by-page extraction here, but we have merged_text
        extraction_dict = {
            "num_pages": doc.num_pages,
            "merged_text": doc.merged_text,
            "pages": [],  # Not stored in DB, would need to reconstruct if needed
            "normalized_pages": [],
        }
        
        # Create LLMAnalysisResult-like structure with all available metadata
        llm_analysis_dict = {
            "instruction": extraction_metadata.get("llm_instruction", instruction),
            "raw_response": extraction_metadata.get("llm_raw_response", ""),
            "structured": {
                **llm_structured,
                "tables": extraction_metadata.get("tables", []),
                "kv_pairs": extraction_metadata.get("kv_pairs", []),
                "headings": extraction_metadata.get("headings", []),
                "detected_tickers": extraction_metadata.get("detected_tickers", []),
                "raw_dates_found": extraction_metadata.get("raw_dates_found", []),
            },
        }
        
        result["action"] = "extraction_completed"
        result["detail"] = {
            "document_id": document_id,
            "extraction": extraction_dict,
            "llm_analysis": llm_analysis_dict,
        }
        
        return result
    except Exception as e:
        result["action"] = "extraction_failed"
        result["detail"]["error"] = str(e)
        return result
    # Note: Don't delete file_path here - it's now stored in the database
