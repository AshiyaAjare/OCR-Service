from typing import List, Optional
from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    page_number: int = Field(..., description="1-based index of the page")
    text_pdf: str = Field("", description="Text extracted via PDF parser")
    text_ocr: str = Field("", description="Text extracted via OCR from page image")


class ExtractionResult(BaseModel):
    num_pages: int
    pages: List[PageExtraction]
    merged_text: str = Field(
        "", description="All page texts (PDF + OCR) concatenated"
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
