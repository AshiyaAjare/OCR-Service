import asyncio
from typing import List, Optional, Dict, Any
import logging

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
from app.services.ollama_client import call_ollama_mistral, extract_json_from_text
from app.services.fallback_extractors import apply_fallback_extractors
from app.tasks.indexing_tasks import index_document_task

import json
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Maximum characters to send to LLM (approximate token limit: ~100k chars ≈ 25k tokens)
MAX_LLM_INPUT_CHARS = 100000


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

def truncate_text_for_llm(text: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """
    Truncate text to prevent token limit issues.
    Tries to truncate at sentence boundaries when possible.
    """
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    # Prefer truncating at paragraph boundary, then sentence
    if last_newline > max_chars * 0.8:  # If we can keep 80% of content
        return text[:last_newline] + "\n\n[Text truncated due to length...]"
    elif last_period > max_chars * 0.8:
        return text[:last_period + 1] + "\n\n[Text truncated due to length...]"
    else:
        return truncated + "\n\n[Text truncated due to length...]"


async def _extract_with_llm(file_path: str, instruction: str, source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Internal helper:
    - runs dual extraction (extract_pdf_dual)
    - calls Ollama/Mistral with the merged text (truncated if needed)
    - uses robust JSON extraction and fallback extractors if LLM fails
    """
    extraction = await extract_pdf_dual(file_path)

    # Truncate text if too long to prevent token limit issues
    truncated_text = truncate_text_for_llm(extraction.merged_text)
    was_truncated = len(extraction.merged_text) > len(truncated_text)

    llm_prompt = (
        f"{instruction}\n\n"
        "Here is the text extracted from the PDF (both direct parsing and OCR):\n\n"
        f"{truncated_text}"
    )

    llm_response: Optional[str] = None
    llm_structured: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None

    try:
        llm_response = await call_ollama_mistral(llm_prompt, temperature=0.0, top_p=0.1)
        
        # Try robust JSON extraction
        llm_structured = extract_json_from_text(llm_response)
        
        if llm_structured is None:
            parse_error = "Failed to extract valid JSON from LLM response"
            logger.warning(f"JSON extraction failed. Raw response: {llm_response[:500]}...")
    except Exception as e:
        parse_error = str(e)
        logger.exception(f"Error calling LLM: {e}")

    # If LLM extraction failed, use fallback extractors
    if llm_structured is None:
        logger.info("Using fallback extractors due to LLM failure")
        llm_structured = apply_fallback_extractors(extraction.merged_text, source_url)
        # Mark that fallback was used
        llm_structured['_fallback_used'] = True
    else:
        llm_structured['_fallback_used'] = False
    
    # Ensure consistent structure: both broker_estimate and broker_estimate_detail should be present
    # If LLM returned broker_estimate_detail but not broker_estimate, extract value from detail
    if llm_structured:
        broker_estimate_detail = llm_structured.get("broker_estimate_detail")
        broker_estimate = llm_structured.get("broker_estimate")
        
        # If we have detail but no top-level estimate, extract value from detail
        if broker_estimate_detail and isinstance(broker_estimate_detail, dict) and broker_estimate is None:
            llm_structured["broker_estimate"] = broker_estimate_detail.get("value")
        
        # If we have top-level estimate but no detail, create minimal detail structure
        if broker_estimate is not None and not broker_estimate_detail:
            llm_structured["broker_estimate_detail"] = {"value": broker_estimate}
        
        # Ensure detail has value if it exists
        if broker_estimate_detail and isinstance(broker_estimate_detail, dict):
            if "value" not in broker_estimate_detail and broker_estimate is not None:
                broker_estimate_detail["value"] = broker_estimate

    return {
        "extraction": extraction,
        "llm_instruction": instruction,
        "llm_raw_response": llm_response,
        "llm_structured": llm_structured,
        "llm_parse_error": parse_error,
        "text_was_truncated": was_truncated,
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
            "- company_name: string or null (the PRIMARY/SUBJECT company this report is about, "
            "NOT competitors or mentioned companies. Look for the company name in document titles, "
            "headers, ticker symbols, and the main subject of the analysis. Examples: 'Asian Paints', "
            "'JSW Steel', 'Steelcase Inc'. Avoid picking competitor names mentioned in passing.)\n"
            "- report_date: string or null in ISO format 'YYYY-MM-DD'\n"
            "- period: string or null (e.g., 'Q2 2025', 'H1 2025')\n"
            "- document_type: string or null (e.g., 'results', 'press_release', "
            "'board_outcome', 'annual_report')\n"
            "- source_domain: string or null (e.g., 'bseindia.com')\n"
            "- language: string or null (e.g., 'en')\n"
            "- ocr_confidence: number or null between 0 and 1 (estimate if needed)\n"
            "- ingestion_method: string or null (e.g., 'pdf', 'web_scrape')\n"
            "- broker_estimate: number or null (broker's equity target price per share if mentioned in the document)\n"
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
            result = asyncio.run(_extract_with_llm(file_path, instruction, source_url))
            llm_error = result.get("llm_parse_error")
        except Exception as e:
            # If LLM (Ollama) is unavailable, fall back to extraction without LLM
            llm_error = str(e)
            logger.exception(f"LLM call failed, using fallback extractors: {e}")
            extraction = asyncio.run(extract_pdf_dual(file_path))
            # Use fallback extractors
            llm_structured = apply_fallback_extractors(extraction.merged_text, source_url)
            llm_structured['_fallback_used'] = True
            result = {
                "extraction": extraction,
                "llm_instruction": instruction,
                "llm_raw_response": None,
                "llm_structured": llm_structured,
                "llm_parse_error": llm_error,
                "text_was_truncated": False,
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

        # Extract broker_estimate_detail and broker_estimate from LLM structured data
        # Both should be present in llm_structured (from LLM or fallback)
        broker_estimate_detail = llm_structured.get("broker_estimate_detail")
        broker_est = llm_structured.get("broker_estimate")
        
        # Ensure broker_estimate_detail is a dict if we have any broker estimate data
        if not broker_estimate_detail and broker_est is not None:
            broker_estimate_detail = {"value": broker_est}
        elif broker_estimate_detail and not isinstance(broker_estimate_detail, dict):
            # If it's not a dict, create one with the value
            broker_estimate_detail = {"value": broker_est} if broker_est is not None else None
        
        # Prefer value from broker_estimate_detail if available, otherwise use broker_estimate
        broker_est_value = None
        if broker_estimate_detail and isinstance(broker_estimate_detail, dict):
            broker_est_value = broker_estimate_detail.get("value")
        elif broker_est is not None:
            broker_est_value = broker_est
        
        # Set the numeric value in the DB column
        if broker_est_value is not None:
            try:
                doc.broker_estimate = float(broker_est_value)
                # Ensure broker_estimate_detail has the value
                if broker_estimate_detail:
                    broker_estimate_detail["value"] = doc.broker_estimate
                else:
                    broker_estimate_detail = {"value": doc.broker_estimate}
            except (TypeError, ValueError):
                doc.broker_estimate = None
                broker_estimate_detail = None
        
        # Fill missing fields in broker_estimate_detail using available info
        if broker_estimate_detail and isinstance(broker_estimate_detail, dict):
            # Fill missing as_of_date from report_date if available
            if not broker_estimate_detail.get("as_of_date") and doc.report_date:
                broker_estimate_detail["as_of_date"] = doc.report_date.strftime("%Y-%m-%d")
            
            # Fill missing currency (default to INR for Indian context)
            if not broker_estimate_detail.get("currency"):
                broker_estimate_detail["currency"] = "INR"
            
            # Fill missing estimate_type (default to target_price)
            if not broker_estimate_detail.get("estimate_type"):
                broker_estimate_detail["estimate_type"] = "target_price"
            
            # Try to infer broker_name from source_domain if missing
            if not broker_estimate_detail.get("broker_name") and doc.source_domain:
                # Map common domains to broker names
                domain_to_broker = {
                    "icicidirect.com": "ICICI Securities",
                    "icicisecurities.com": "ICICI Securities",
                    "hdfcsec.com": "HDFC Securities",
                    "kotaksecurities.com": "Kotak Securities",
                    "axisdirect.com": "Axis Securities",
                    "motilaloswal.com": "Motilal Oswal",
                    "edelweiss.in": "Edelweiss",
                    "avendusspark.com": "Avendus Spark",
                }
                domain_lower = doc.source_domain.lower()
                for domain, broker in domain_to_broker.items():
                    if domain in domain_lower:
                        broker_estimate_detail["broker_name"] = broker
                        break
            
            # If still missing, try to extract from document text
            if not broker_estimate_detail.get("broker_name") and extraction.merged_text:
                from app.services.fallback_extractors import extract_broker_name_from_text
                extracted_broker = extract_broker_name_from_text(extraction.merged_text)
                if extracted_broker:
                    broker_estimate_detail["broker_name"] = extracted_broker
            
            # Save broker_name to database column if available
            if broker_estimate_detail.get("broker_name"):
                doc.broker_name = broker_estimate_detail.get("broker_name")

        # ingestion_method / source_domain may already be set from meta/source_url
        if llm_structured.get("ingestion_method"):
            doc.ingestion_method = llm_structured.get("ingestion_method")

        # Build rich extraction_metadata JSON
        fallback_used = llm_structured.get("_fallback_used", False)
        doc.extraction_metadata = {
            # Existing keys
            "meta": meta,
            "llm_instruction": result["llm_instruction"],
            "llm_raw_response": result["llm_raw_response"],
            "llm_error": llm_error or result.get("llm_parse_error"),
            "llm_fallback_used": fallback_used,
            "text_was_truncated": result.get("text_was_truncated", False),
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
                "broker_estimate": doc.broker_estimate,
                "broker_name": doc.broker_name,
            },
            # Store richer broker estimate details
            "broker_estimate_detail": broker_estimate_detail,
        }
        doc.is_processed = True
        doc.processing_status = "completed"
        db.commit()
        try:
            index_document_task.delay(document_id)
        except Exception as e:
            logger.exception("Failed to enqueue indexing task for %s: %s", document_id, e)
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
