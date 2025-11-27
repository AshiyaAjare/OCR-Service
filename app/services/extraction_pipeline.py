import asyncio
from typing import List

from app.models.schemas import ExtractionResult, PageExtraction
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
    merged_fragments: List[str] = []

    for idx in range(num_pages):
        pdf_text = pdf_text_pages[idx] if idx < len(pdf_text_pages) else ""
        ocr_text = ocr_text_pages[idx] if idx < len(ocr_text_pages) else ""

        page = PageExtraction(
            page_number=idx + 1,
            text_pdf=pdf_text,
            text_ocr=ocr_text,
        )
        pages.append(page)

        merged_fragments.append(f"[PAGE {idx+1} PDF]\n{pdf_text}\n")
        merged_fragments.append(f"[PAGE {idx+1} OCR]\n{ocr_text}\n")

    merged_text = "\n".join(merged_fragments)

    return ExtractionResult(
        num_pages=num_pages,
        pages=pages,
        merged_text=merged_text,
    )
