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

import json
from urllib.parse import urlparse


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

    llm_structured: Optional[Dict[str, Any]] = None
    try:
        llm_structured = json.loads(llm_response)
    except (TypeError, ValueError):
        llm_structured = None

    return {
        "extraction": extraction,
        "llm_instruction": instruction,
        "llm_raw_response": llm_response,
        "llm_structured": llm_structured,
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
        (
            "You are analyzing a financial or regulatory PDF. "
            "Return a single JSON object with these top-level keys:\n"
            "- ticker: string or null (e.g., 'STEELCAS')\n"
            "- company_name: string or null\n"
            "- report_date: string or null in ISO format 'YYYY-MM-DD'\n"
            "- period: string or null (e.g., 'Q2 2025', 'H1 2025')\n"
            "- document_type: string or null (e.g., 'results', 'press_release', "
            "'board_outcome', 'annual_report')\n"
            "- source_domain: string or null (e.g., 'bseindia.com')\n"
            "- language: string or null (e.g., 'en')\n"
            "- ocr_confidence: number or null between 0 and 1 (estimate if needed)\n"
            "- ingestion_method: string or null (e.g., 'pdf', 'web_scrape')\n"
            "- tables: array of objects, each with fields like "
            "{'page': int, 'title': string|null, 'summary': string, 'headers': [...]} \n"
            "- kv_pairs: array or object capturing key metrics, for example "
            "[{'key': 'net_sales', 'value': 11060.31, 'unit': 'INR_lakhs', "
            "'period': '30-09-2025'}]\n"
            "- headings: array of objects like "
            "{'page': int, 'heading': string, 'level': int|null}\n"
            "- detected_tickers: array of objects like "
            "{'ticker': string, 'confidence': number 0..1}\n"
            "- raw_dates_found: array of objects like "
            "{'raw': string, 'normalized': string|null}\n\n"
            "Only output valid JSON. Do not include explanations or comments."
        ),
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
    ingestion_method = meta.get("ingestion_method", "pdf")
    doc.ingestion_method = ingestion_method

    source_url = meta.get("source_url")
    if source_url:
        try:
            parsed = urlparse(source_url)
            doc.source_domain = parsed.netloc or None
        except Exception:
            doc.source_domain = None
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
        try:
            result = asyncio.run(_extract_with_llm(file_path, instruction))
            llm_error = None
        except Exception as e:
            # If LLM (Ollama) is unavailable, fall back to extraction without LLM
            llm_error = str(e)
            extraction = asyncio.run(extract_pdf_dual(file_path))
            result = {
                "extraction": extraction,
                "llm_instruction": instruction,
                "llm_raw_response": None,
                "llm_structured": None,
            }

        extraction = result["extraction"]
        llm_structured = result.get("llm_structured") or {}

        # Update document record with results
        doc.num_pages = extraction.num_pages
        doc.merged_text = extraction.merged_text

        doc.ticker = llm_structured.get("ticker")
        doc.company_name = llm_structured.get("company_name")
        report_date_str = llm_structured.get("report_date")
        if report_date_str:
            from datetime import date
            try:
                parts = report_date_str.split("-")
                if len(parts) == 3:
                    y, m, d = map(int, parts)
                    doc.report_date = date(y, m, d)
            except Exception:
                doc.report_date = None
        doc.period = llm_structured.get("period")
        doc.document_type = llm_structured.get("document_type") or doc.document_type
        # Prefer LLM language if present
        doc.language = llm_structured.get("language") or doc.language

        ocr_conf = llm_structured.get("ocr_confidence")
        try:
            doc.ocr_confidence = float(ocr_conf) if ocr_conf is not None else doc.ocr_confidence
        except (TypeError, ValueError):
            pass

        # ingestion_method / source_domain may already be set from meta/source_url
        if llm_structured.get("ingestion_method"):
            doc.ingestion_method = llm_structured.get("ingestion_method")

        # Build rich extraction_metadata JSON
        doc.extraction_metadata = {
            # Existing keys
            "meta": meta,
            "llm_instruction": result["llm_instruction"],
            "llm_raw_response": result["llm_raw_response"],
            "llm_error": llm_error,
            "llm_fallback_used": llm_error is not None,
            "tables": llm_structured.get("tables", []),
            "kv_pairs": llm_structured.get("kv_pairs", []),
            "headings": llm_structured.get("headings", []),
            "detected_tickers": llm_structured.get("detected_tickers", []),
            "raw_dates_found": llm_structured.get("raw_dates_found", []),
            "scalar_metadata": {
                "ticker": doc.ticker,
                "company_name": doc.company_name,
                "report_date": report_date_str,
                "period": doc.period,
                "document_type": doc.document_type,
                "source_domain": doc.source_domain,
                "language": doc.language,
                "ocr_confidence": doc.ocr_confidence,
                "ingestion_method": doc.ingestion_method,
            },
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
