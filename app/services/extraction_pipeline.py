import asyncio
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from app.models.schemas import (
    ExtractionResult,
    PageExtraction,
    PageExtractionNormalized,
    NormalizedText,
)
from app.services.text_normalizer import normalize_text_to_lines
from app.services.pdf_text_extractor import extract_text_from_pdf_pages
from app.services.pdf_image_extractor import render_pdf_to_images
from app.services.ocr_service import run_ocr_on_images
from app.database import SessionLocal
from app.models.db_models import ExtractedDocumentModel
from app.services.ollama_client import call_ollama_mistral


async def extract_pdf_dual(file_path: str) -> ExtractionResult:
    """
    Orchestrates dual extraction:
    - Text via PDF parsing
    - Text via OCR on rendered images

    The PDF text & rendering to images are run in parallel using asyncio.run_in_executor.
    """

    loop = asyncio.get_event_loop()
    
    # Run PDF text parsing and page rendering concurrently
    text_task = loop.run_in_executor(None, extract_text_from_pdf_pages, file_path)
    images_task = loop.run_in_executor(None, render_pdf_to_images, file_path)

    pdf_text_pages, images = await asyncio.gather(text_task, images_task)

    # Run OCR on all rendered images (can be heavy)
    ocr_text_pages = await loop.run_in_executor(None, run_ocr_on_images, images, "eng")

    num_pages = max(len(pdf_text_pages), len(ocr_text_pages))

    pages: List[PageExtraction] = []

    normalized_pages: List[PageExtractionNormalized] = []

    merged_fragments: List[str] = []

    for idx in range(num_pages):

        pdf_text = pdf_text_pages[idx] if idx < len(pdf_text_pages) else ""

        ocr_text = ocr_text_pages[idx] if idx < len(ocr_text_pages) else ""

        # Raw page info (what you already have)
        page = PageExtraction(
            page_number=idx + 1,
            text_pdf=pdf_text,
            text_ocr=ocr_text,
        )
        pages.append(page)

        # Normalized (verification view)
        norm_pdf_lines = normalize_text_to_lines(pdf_text)
        norm_ocr_lines = normalize_text_to_lines(ocr_text)
        normalized_pages.append(
            PageExtractionNormalized(
                page_number=idx + 1,
                pdf=NormalizedText(lines=norm_pdf_lines),
                ocr=NormalizedText(lines=norm_ocr_lines),
            )
        )

        # Keep merged_text for LLM use
        merged_fragments.append(f"[PAGE {idx+1} PDF]\n{pdf_text}\n")
        merged_fragments.append(f"[PAGE {idx+1} OCR]\n{ocr_text}\n")

    merged_text = "\n".join(merged_fragments)

    return ExtractionResult(
        num_pages=num_pages,
        pages=pages,
        merged_text=merged_text,
        normalized_pages=normalized_pages,
    )

async def _extract_with_llm(file_path: str, instruction: str) -> Dict[str, Any]:
    """
    Internal helper:
    - runs dual extraction (extract_pdf_dual)
    - calls Ollama/Mistral with the merged text
    """
    extraction = await extract_pdf_dual(file_path)

    llm_prompt = (
        f"{instruction}\n\n"
        "Here is the text extracted from the PDF (both direct parsing and OCR):\n\n"
        f"{extraction.merged_text}"
    )

    llm_response = await call_ollama_mistral(llm_prompt)

    return {
        "extraction": extraction,
        "llm_instruction": instruction,
        "llm_raw_response": llm_response,
    }


def process_pdf_from_file(
    file_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Synchronous wrapper used by link_resolver_service.

    - Accepts a local PDF path.
    - Optionally takes a meta dict (e.g. source_url, source_email_id).
    - Persists an ExtractedDocumentModel row with status updates.
    - Runs the dual extraction coroutine and returns the document ID.
    """
    meta = meta or {}
    instruction = meta.get(
        "llm_instruction",
        "Summarize the key points and return a short JSON with fields "
        "company_name, period, key_financials, dividends, notes.",
    )

    db: Session = SessionLocal()
    doc = ExtractedDocumentModel(
        source_email_id=meta.get("source_email_id"),
        source_url=meta.get("source_url"),
        file_path=file_path,
        file_type="pdf",
        is_processed=False,
        processing_status="processing",
    )
    try:
        db.add(doc)
        db.commit()
        db.refresh(doc)
        document_id = doc.id
    except Exception:
        db.rollback()
        db.close()
        raise

    try:
        # Run dual extraction + LLM, mirroring /extract-with-llm
        result = asyncio.run(_extract_with_llm(file_path, instruction))
        extraction = result["extraction"]

        # Update document record with results
        doc.num_pages = extraction.num_pages
        doc.merged_text = extraction.merged_text
        doc.extraction_metadata = {
            "meta": meta,
            "llm_instruction": result["llm_instruction"],
            "llm_raw_response": result["llm_raw_response"],
        }
        doc.is_processed = True
        doc.processing_status = "completed"
        db.commit()
    except Exception as e:
        # Mark as failed
        doc.processing_status = "failed"
        doc.is_processed = False
        doc.extraction_metadata = {"meta": meta, "error": str(e)}
        db.commit()
        raise
    finally:
        db.close()

    return document_id
