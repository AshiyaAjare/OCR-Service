# app/models/db_models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.database import Base
from app.config import settings


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
    
    # Extraction results
    num_pages = Column(Integer, nullable=True)
    merged_text = Column(Text, nullable=True)
    extraction_metadata = Column(JSON, nullable=True)  # Store full extraction result as JSON
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_status = Column(String(50), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

