import asyncio
from typing import List

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
