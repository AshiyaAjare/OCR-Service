# app/models/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    page_number: int = Field(..., description="1-based index of the page")
    text_pdf: str = Field("", description="Raw text extracted via PDF parser")
    text_ocr: str = Field("", description="Raw text extracted via OCR from page image")


class NormalizedText(BaseModel):
    lines: List[str] = Field(
        default_factory=list,
        description="Cleaned, line-wise representation with no \\n characters.",
    )


class PageExtractionNormalized(BaseModel):
    page_number: int
    pdf: NormalizedText
    ocr: NormalizedText


class ExtractionResult(BaseModel):
    num_pages: int
    pages: List[PageExtraction]
    merged_text: str = Field(
        "", description="All page texts (PDF + OCR) concatenated"
    )
    normalized_pages: List[PageExtractionNormalized] = Field(
        default_factory=list,
        description="Verification-friendly, nested view of the extracted text",
    )


class LLMAnalysisResult(BaseModel):
    instruction: str
    raw_response: str


class FullAnalysisResponse(BaseModel):
    extraction: ExtractionResult
    llm_analysis: LLMAnalysisResult


class LLMInstruction(BaseModel):
    instruction: str = Field(
        ...,
        description=(
            "What you want Mistral to do with the extracted text. "
            "E.g. 'Summarize key financial metrics in JSON with keys revenue, profit, eps.'"
        ),
    )
