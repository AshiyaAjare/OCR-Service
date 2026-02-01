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


# Schemas for House vs Broker Analysis API
class HouseVsBrokerAnalysisItem(BaseModel):
    """Individual company analysis result"""
    company_id: int = Field(..., description="Company ID")
    company_name: str = Field(..., description="Company name")
    house_value: Optional[float] = Field(None, description="House value (fv_cmp) from usermanagement_companylisting")
    broker_estimate: Optional[float] = Field(None, description="Broker estimate from ocr_extracted_document")
    difference: Optional[float] = Field(None, description="Difference: broker_estimate - house_value")
    percentage_difference: Optional[float] = Field(None, description="Percentage difference: ((broker_estimate - house_value) / house_value) * 100")


class HouseVsBrokerAnalysisResponse(BaseModel):
    """Response for house vs broker analysis API"""
    results: List[HouseVsBrokerAnalysisItem] = Field(..., description="List of analysis results")
    meta: dict = Field(..., description="Metadata about the response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "company_id": 1,
                        "company_name": "Example Company",
                        "house_value": 1000.0,
                        "broker_estimate": 1100.0,
                        "difference": 100.0,
                        "percentage_difference": 10.0
                    }
                ],
                "meta": {
                    "count": 1
                }
            }
        }
