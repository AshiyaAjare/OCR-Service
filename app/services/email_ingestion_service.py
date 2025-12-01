# app/services/email_ingestion_service.py

import imaplib
import email
from email.header import decode_header
from email.message import Message
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re

from app.config import settings


# ---------- 1. DATA MODELS ----------

@dataclass
class IngestedEmail:
    message_id: str
    subject: str
    sender: str
    date: str
    body_text: str
    body_html: Optional[str]
    urls: List[str]


# ---------- 2. HELPER FUNCTIONS ----------

def _decode_header(value: Optional[str]) -> str:
    """Decode MIME-encoded header strings into plain text."""
    if not value:
        return ""
    decoded_parts = decode_header(value)
    pieces = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            try:
                pieces.append(part.decode(enc or "utf-8", errors="ignore"))
            except LookupError:
                pieces.append(part.decode("utf-8", errors="ignore"))
        else:
            pieces.append(part)
    return "".join(pieces).strip()


def _extract_urls_from_text(text: str) -> List[str]:
    """Extract web/HTTP(S) URLs from a text blob."""
    if not text:
        return []
    # Simple URL regex (good enough for PoC)
    url_pattern = r"https?://[^\s<>\"']+"
    return re.findall(url_pattern, text)


def _get_email_bodies(msg: Message) -> Tuple[str, Optional[str]]:
    """
    Return (plain_text_body, html_body) from an email.message.Message.
    If only HTML is present, we'll still return it as html and attempt a plain-text fallback.
    """
    text_body = ""
    html_body: Optional[str] = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", "").lower()

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            try:
                payload = part.get_payload(decode=True)
            except Exception:
                payload = None

            if not payload:
                continue

            charset = part.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="ignore")
            except LookupError:
                decoded = payload.decode("utf-8", errors="ignore")

            if content_type == "text/plain":
                text_body += decoded + "\n"
            elif content_type == "text/html":
                html_body = (html_body or "") + decoded
    else:
        # Single part message
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="ignore")
        except LookupError:
            decoded = payload.decode("utf-8", errors="ignore")

        if content_type == "text/plain":
            text_body = decoded
        elif content_type == "text/html":
            html_body = decoded

    # If no plain text but we have HTML, we can return a very rough text fallback
    if not text_body and html_body:
        # Strip very roughly, better to use BeautifulSoup later if needed
        text_body = re.sub("<[^>]+>", " ", html_body)
        text_body = re.sub(r"\s+", " ", text_body).strip()

    return text_body.strip(), html_body


# ---------- 3. MAIN SERVICE CLASS ----------

class GmailIngestionService:
    """
    Service responsible for connecting to Gmail via IMAP,
    fetching emails, and returning them in a structured format.

    This PoC version:
    - Connects to INBOX
    - Optionally filters by FROM and/or SUBJECT keywords
    - Returns the N most recent matching emails
    """

    def __init__(
        self,
        imap_host: str = None,
        email_address: str = None,
        app_password: str = None,
    ):
        self.imap_host = imap_host or settings.gmail_imap_host
        self.email_address = email_address or settings.gmail_email
        self.app_password = app_password or settings.gmail_app_password

    def _connect(self) -> imaplib.IMAP4_SSL:
        """Create and authenticate an IMAP connection."""
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host)
            mail.login(self.email_address, self.app_password)
            return mail
        except imaplib.IMAP4.error as e:
            raise ConnectionError(f"Failed to connect to IMAP server: {e}") from e
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to IMAP: {e}") from e

    def fetch_recent_emails(
        self,
        max_emails: int = 20,
        from_filter: Optional[str] = None,
        subject_contains: Optional[str] = None,
        only_unseen: bool = False,
    ) -> List[IngestedEmail]:
        """
        Fetch most recent emails matching the provided filters.

        - from_filter: filter by sender email (e.g. 'noreply@broker.com')
        - subject_contains: filter emails whose subject contains this text
        - only_unseen: if True, restrict to unseen messages only
        """
        mail = self._connect()
        try:
            mail.select("INBOX")

            # Build IMAP search criteria properly
            # IMAP search requires a single string with space-separated criteria
            search_parts = []
            if only_unseen:
                search_parts.append("UNSEEN")
            else:
                search_parts.append("ALL")
            
            if from_filter:
                # IMAP FROM search expects the email address without quotes in the search string
                search_parts.append(f'FROM "{from_filter}"')
            
            if subject_contains:
                # IMAP SUBJECT search expects the text without quotes in the search string
                search_parts.append(f'SUBJECT "{subject_contains}"')

            # Build IMAP search query string
            search_query = " ".join(search_parts)
            status, data = mail.search(None, search_query)

            if status != "OK":
                return []

            # data[0] is bytes, split it and convert to list of bytes
            all_ids = data[0].split()
            if not all_ids:
                return []
            
            # Get only the most recent emails (IMAP returns oldest first, so take last N)
            email_ids = all_ids[-max_emails:] if len(all_ids) > max_emails else all_ids

            ingested: List[IngestedEmail] = []
            for eid in email_ids:
                # eid is bytes, mail.fetch expects bytes or string
                status, msg_data = mail.fetch(eid, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject = _decode_header(msg.get("Subject"))
                sender = _decode_header(msg.get("From"))
                date = msg.get("Date", "")

                message_id = msg.get("Message-ID") or msg.get("Message-Id") or ""

                body_text, body_html = _get_email_bodies(msg)
                combined_text = body_text + "\n" + (body_html or "")
                urls = list(dict.fromkeys(_extract_urls_from_text(combined_text)))  # Deduplicate while preserving order

                ingested.append(
                    IngestedEmail(
                        message_id=message_id,
                        subject=subject,
                        sender=sender,
                        date=date,
                        body_text=body_text,
                        body_html=body_html,
                        urls=urls,
                    )
                )

            return ingested
        finally:
            try:
                mail.close()
            except Exception:
                pass
            mail.logout()
