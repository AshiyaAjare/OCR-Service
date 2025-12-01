# app/models/__init__.py

# Export Pydantic schemas (for API)
from app.models.schemas import (
    PageExtraction,
    NormalizedText,
    PageExtractionNormalized,
    ExtractionResult,
    LLMAnalysisResult,
    FullAnalysisResponse,
    LLMInstruction,
)

# Export database models (for SQLAlchemy)
from app.models.db_models import (
    PrefixedBase,
    IngestedEmailModel,
    EmailAttachmentModel,
    ExtractedDocumentModel,
)

__all__ = [
    # Pydantic schemas
    "PageExtraction",
    "NormalizedText",
    "PageExtractionNormalized",
    "ExtractionResult",
    "LLMAnalysisResult",
    "FullAnalysisResponse",
    "LLMInstruction",
    # Database models
    "PrefixedBase",
    "IngestedEmailModel",
    "EmailAttachmentModel",
    "ExtractedDocumentModel",
]

