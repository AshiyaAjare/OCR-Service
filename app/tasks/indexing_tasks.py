# app/tasks/indexing_tasks.py
import logging
from datetime import datetime

from app.celery_app import celery
from app.database import SessionLocal
from app.models.db_models import ExtractedDocumentModel
from app.services.index_pdf_to_pgvector import index_merged_text

logger = logging.getLogger(__name__)

@celery.task(bind=True, max_retries=3, acks_late=True, soft_time_limit=600)
def index_document_task(self, document_id: int):
    """
    Worker task to index merged_text for document_id into pgvector.
    - idempotent: checks vector_indexed flag first
    - retries on error
    """
    db = SessionLocal()
    try:
        doc = db.query(ExtractedDocumentModel).get(document_id)
        if doc is None:
            logger.warning("Document %s not found, skipping", document_id)
            return {"status": "missing"}

        # idempotency: skip if already indexed
        if getattr(doc, "vector_indexed", False):
            logger.info("Document %s already indexed, skipping", document_id)
            return {"status": "already_indexed"}

        merged = doc.merged_text or ""
        # If merged_text is large, ensure index_merged_text handles truncation etc.
        index_merged_text(merged, document_id=str(document_id))

        # update flags
        doc.vector_indexed = True
        doc.vector_indexed_at = datetime.utcnow()
        db.commit()
        logger.info("Document %s indexed successfully", document_id)
        return {"status": "ok"}
    except Exception as exc:
        db.rollback()
        logger.exception("Indexing failed for document %s: %s", document_id, exc)
        # retry with exponential backoff
        raise self.retry(exc=exc, countdown=60)
    finally:
        db.close()
