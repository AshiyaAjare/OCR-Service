# app/api/routes_email.py

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.email_ingestion_service import (
    GmailIngestionService,
    persist_ingested_emails,
)
from app.services.link_resolver_service import resolve_and_process_link

router = APIRouter(prefix="/api/v1/email", tags=["email"])


@router.post("/ingest", summary="Fetch recent emails and persist them")
def ingest_emails(
    max_emails: int = 3,
    only_unseen: bool = True,
    from_filter: Optional[str] = None,
    subject_contains: Optional[str] = None,
    db: Session = Depends(get_db),
):
    service = GmailIngestionService()
    emails = service.fetch_recent_emails(
        max_emails=max_emails,
        from_filter=from_filter,
        subject_contains=subject_contains,
        only_unseen=only_unseen,
    )

    created = persist_ingested_emails(emails, db=db)
    return {
        "fetched": len(emails),
        "created": created,
    }

@router.post("/resolve")
def resolve_link(url: str):
    result = resolve_and_process_link(
        url,
        meta={
            "source": "api-test",
            "source_url": url,
        },
    )
    return result