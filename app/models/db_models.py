# app/models/db_models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func, text
from app.database import Base
from app.config import settings
from sqlalchemy import Date, Float


# Base class with table prefix support
class PrefixedBase(Base):
    """
    Base class that automatically prefixes table names to avoid conflicts
    with Django project's tables in the same database.
    """
    __abstract__ = True


class IngestedEmailModel(PrefixedBase):
    """
    Database model for ingested emails.
    Table name will be: ocr_ingested_email (with default prefix)
    """
    __tablename__ = f"{settings.DATABASE_TABLE_PREFIX}ingested_email"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String(500), unique=True, index=True, nullable=False)
    subject = Column(Text, nullable=True)
    sender = Column(String(500), nullable=False)
    date = Column(String(100), nullable=True)
    body_text = Column(Text, nullable=True)
    body_html = Column(Text, nullable=True)
    urls = Column(JSON, nullable=True)  # Store list of URLs as JSON
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_status = Column(String(50), nullable=True)  # e.g., 'pending', 'processing', 'completed', 'failed'


class EmailAttachmentModel(PrefixedBase):
    """
    Database model for email attachments (if you want to store attachment metadata).
    Table name will be: ocr_email_attachment
    """
    __tablename__ = f"{settings.DATABASE_TABLE_PREFIX}email_attachment"

    id = Column(Integer, primary_key=True, index=True)
    email_id = Column(Integer, nullable=False, index=True)  # Foreign key to ingested_email
    filename = Column(String(500), nullable=False)
    content_type = Column(String(200), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_path = Column(String(1000), nullable=True)  # Path to stored file if downloaded
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExtractedDocumentModel(PrefixedBase):
    """
    Database model for extracted documents (PDFs, etc.).
    Table name will be: ocr_extracted_document
    """
    __tablename__ = f"{settings.DATABASE_TABLE_PREFIX}extracted_document"

    id = Column(Integer, primary_key=True, index=True)
    source_email_id = Column(Integer, nullable=True, index=True)  # Optional link to email
    source_url = Column(String(1000), nullable=True)  # If extracted from URL
    file_path = Column(String(1000), nullable=False)
    file_type = Column(String(50), nullable=False)  # e.g., 'pdf', 'html'
    
    # New top-level metadata columns (nullable, some indexed for querying)
    ticker = Column(String(100), nullable=True, index=True)  # e.g., STEELCAS
    company_name = Column(String(500), nullable=True, index=True)
    broker_name = Column(String(500), nullable=True, index=True)  # e.g., ICICI Securities
    report_date = Column(Date, nullable=True)  # YYYY-MM-DD
    period = Column(String(100), nullable=True)  # e.g., Q2 2025, H1 2025
    document_type = Column(String(100), nullable=True, index=True)  # results, press_release, etc.
    source_domain = Column(String(255), nullable=True, index=True)  # e.g., bseindia.com
    language = Column(String(10), nullable=True)  # e.g., en
    ocr_confidence = Column(Float, nullable=True)  # 0..1
    ingestion_method = Column(String(50), nullable=True, index=True)  # pdf, web_scrape

    # Extraction results
    num_pages = Column(Integer, nullable=True)
    merged_text = Column(Text, nullable=True)
    extraction_metadata = Column(JSON, nullable=True)  # Store full extraction result as JSON
    
    # Broker Estimate
    # DB column: stores the numeric equity target price value for querying/indexing
    # Rich details stored in extraction_metadata["broker_estimate_detail"] with:
    #   - value: float (same as broker_estimate column)
    #   - estimate_type: str (e.g., "target_price")
    #   - currency: str (e.g., "INR")
    #   - as_of_date: str (e.g., "2025-12-04")
    #   - broker_name: str (e.g., "ICICI Securities")
    #   - rating: str (e.g., "HOLD")
    #   - horizon: str (e.g., "12M (I-Sec rating framework)")
    broker_estimate = Column(Float, nullable=True, index=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_status = Column(String(50), nullable=True)

    # Indexing status
    vector_indexed = Column(Boolean, server_default=text('false'), nullable=False, index=True)
    vector_indexed_at = Column(DateTime(timezone=True), nullable=True)

    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

