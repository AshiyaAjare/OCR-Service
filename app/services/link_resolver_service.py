# app/services/link_resolver_service.py
"""
Link Resolver Service

Responsibilities:
- Given a URL, determine whether it is a PDF or a web page.
- If PDF -> call OCR / extraction pipeline.
- If web -> call web-scraping service.
- Provide a small result object that indicates which path was taken and summary metadata.

Notes:
- The service uses HEAD requests to inspect `Content-Type` when possible, falls back to GET if needed,
  and also tries simple URL-extension checks.
- If an HTML page is returned and contains a direct PDF link, the resolver will follow that PDF link.
- Errors are handled gracefully and returned in the result object so callers can log/act accordingly.
"""

import logging
import mimetypes
import os
import tempfile
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

# Import your internal services (adjust import paths if needed)
from app.services import extraction_pipeline  # your existing extraction/orchestrator
# from app.services import ocr_service            # if you call OCR directly
from app.services import web_scraper_service  # your web scraping service (create if not present)

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 10  # secs for requests
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _looks_like_pdf_by_extension(url: str) -> bool:
    """Quick heuristic: url path ends with .pdf (case-insensitive) or has pdf in query param filename."""
    path = urlparse(url).path or ""
    if path.lower().endswith(".pdf"):
        return True
    # Handle cases like /download?file=report.pdf
    lower = url.lower()
    return ".pdf" in lower and ("file=" in lower or "download" in lower)


def _head_content_type(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """Issue HEAD; return Content-Type if available. Falls back to None on failure."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        # Some servers return 405 for HEAD; caller should fallback to GET if absent
        if resp.status_code >= 400:
            return None
        ctype = resp.headers.get("Content-Type")
        return ctype
    except requests.RequestException as e:
        logger.debug("HEAD failed for %s: %s", url, e)
        return None


def _get_content_type_via_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """Perform a lightweight GET (streamed) to get headers (used when HEAD is not reliable)."""
    try:
        resp = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
        ctype = resp.headers.get("Content-Type")
        # Important: close the connection to not keep sockets open
        resp.close()
        return ctype
    except requests.RequestException as e:
        logger.debug("GET (for content-type) failed for %s: %s", url, e)
        return None


def _download_to_tempfile(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Download URL content to a temp file and return absolute path. Caller must delete file when done."""
    resp = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=timeout,
        headers=DEFAULT_HEADERS,
    )
    resp.raise_for_status()
    suffix = ".pdf" if "pdf" in (resp.headers.get("Content-Type") or "").lower() else ""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path


def _find_pdf_link_in_html(base_url: str, html_text: str) -> Optional[str]:
    """Scan HTML and return the first absolute URL pointing to a .pdf (if any)."""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            # normalize relative links
            full = urljoin(base_url, href)
            if _looks_like_pdf_by_extension(full):
                return full
        return None
    except Exception as e:
        logger.debug("Error parsing HTML for pdf links: %s", e)
        return None


def resolve_and_process_link(
    url: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    prefer_stream_download: bool = False,
) -> Dict[str, Any]:
    """
    Main entrypoint.

    Parameters:
    - url: the input link
    - meta: optional metadata (e.g. email_id, message_id, subject, sender) passed downstream
    - prefer_stream_download: if True, always download before handing to extractor (useful when extractor expects a local file)

    Returns:
    {
        "url": url,
        "resolved_type": "pdf" | "web" | "unknown" | "pdf_found_in_html",
        "action": "ocr_called" | "scraper_called" | "skipped" | "failed",
        "detail": {...}  # additional info or error
    }
    """
    meta = meta or {}
    result = {"url": url, "resolved_type": None, "action": None, "detail": {}}

    # 1) Heuristic: extension check
    if _looks_like_pdf_by_extension(url):
        result["resolved_type"] = "pdf"
        document_id = None
        try:
            # Option A: call extraction directly with URL if your pipeline supports URLs
            # e.g. if extraction_pipeline has process_pdf_from_url(url, meta)
            if hasattr(extraction_pipeline, "process_pdf_from_url") and not prefer_stream_download:
                logger.info("PDF detected by extension. Calling extraction_pipeline.process_pdf_from_url")
                document_id = extraction_pipeline.process_pdf_from_url(url, meta=meta)
                result["action"] = "ocr_called"
                result["detail"]["document_id"] = document_id
                return result

            # Option B: download and call local-file based processor
            tmp_path = _download_to_tempfile(url)
            try:
                logger.info("Downloaded PDF to %s -> calling extraction pipeline", tmp_path)
                # Example call - adapt to your implementation:
                if hasattr(extraction_pipeline, "process_pdf_from_file"):
                    document_id = extraction_pipeline.process_pdf_from_file(
                        tmp_path,
                        meta={"source_url": url, **meta},
                    )
                elif hasattr(extraction_pipeline, "process_pdf_bytes"):
                    with open(tmp_path, "rb") as fh:
                        document_id = extraction_pipeline.process_pdf_bytes(
                            fh.read(),
                            meta={"source_url": url, **meta},
                        )
                else:
                    # Fallback if your OCR expects a different function - call ocr_service
                    # from app.services import ocr_service
                    # ocr_service.process_pdf_file(tmp_path, meta=meta)
                    raise NotImplementedError("No compatible PDF entrypoint found in extraction_pipeline")
                result["action"] = "ocr_called"
                if document_id is not None:
                    result["detail"]["document_id"] = document_id
                return result
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.debug("Could not delete temp file %s", tmp_path)
        except Exception as e:
            logger.exception("Error processing PDF URL %s: %s", url, e)
            result["action"] = "failed"
            result["detail"]["error"] = str(e)
            return result

    # 2) Try HEAD to fetch Content-Type
    ctype = _head_content_type(url)
    if not ctype:
        ctype = _get_content_type_via_get(url)
    if ctype:
        ctype_low = ctype.split(";")[0].strip().lower()
        logger.debug("URL %s returned content-type %s", url, ctype_low)
        if ctype_low in ("application/pdf", "application/x-pdf"):
            result["resolved_type"] = "pdf"
            try:
                # download & process similar to above
                tmp_path = _download_to_tempfile(url)
                try:
                    if hasattr(extraction_pipeline, "process_pdf_from_file"):
                        extraction_pipeline.process_pdf_from_file(tmp_path, meta=meta)
                    else:
                        with open(tmp_path, "rb") as fh:
                            extraction_pipeline.process_pdf_bytes(fh.read(), meta=meta)
                    result["action"] = "ocr_called"
                    return result
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        logger.debug("Could not delete temp file %s", tmp_path)
            except Exception as e:
                logger.exception("Error processing PDF URL %s: %s", url, e)
                result["action"] = "failed"
                result["detail"]["error"] = str(e)
                return result
        elif ctype_low.startswith("text/") or ctype_low in ("application/xhtml+xml",):
            # treat as web page -> scrape
            result["resolved_type"] = "web"
            try:
                logger.info("Treating as web page; calling web_scraper_service.scrape_url")
                web_scraper_service.scrape_url(url, meta=meta)
                result["action"] = "scraper_called"
                return result
            except Exception as e:
                logger.exception("Error scraping web URL %s: %s", url, e)
                result["action"] = "failed"
                result["detail"]["error"] = str(e)
                return result

    # 3) As final fallback: fetch the URL body, attempt to parse HTML and look for PDF link
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.exception("Final GET failed for %s: %s", url, e)
        result["action"] = "failed"
        result["resolved_type"] = "unknown"
        result["detail"]["error"] = str(e)
        return result

    # if server returned a pdf in body (rare if header missing)
    body_ct = resp.headers.get("Content-Type", "").lower()
    if "application/pdf" in body_ct:
        result["resolved_type"] = "pdf"
        # save and call pipeline
        try:
            tmp_path = None
            if prefer_stream_download:
                tmp_path = _download_to_tempfile(url)
            else:
                # write resp.content to tempfile
                fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
            try:
                if hasattr(extraction_pipeline, "process_pdf_from_file"):
                    extraction_pipeline.process_pdf_from_file(tmp_path, meta=meta)
                else:
                    with open(tmp_path, "rb") as fh:
                        extraction_pipeline.process_pdf_bytes(fh.read(), meta=meta)
                result["action"] = "ocr_called"
                return result
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.debug("Couldn't remove tmp file %s", tmp_path)
        except Exception as e:
            logger.exception("Error processing PDF content from %s: %s", url, e)
            result["action"] = "failed"
            result["detail"]["error"] = str(e)
            return result

    # if it's HTML, check for PDF link inside. If found -> process that PDF.
    html = resp.text or ""
    pdf_in_html = _find_pdf_link_in_html(url, html)
    if pdf_in_html:
        result["resolved_type"] = "pdf_found_in_html"
        try:
            # Recursively resolve the found PDF link
            return resolve_and_process_link(pdf_in_html, meta=meta, prefer_stream_download=prefer_stream_download)
        except Exception as e:
            logger.exception("Error processing PDF found inside HTML for %s -> %s", url, e)
            result["action"] = "failed"
            result["detail"]["error"] = str(e)
            return result

    # Otherwise treat it as a web page to be scraped
    try:
        result["resolved_type"] = "web"
        web_scraper_service.scrape_url(url, meta=meta)
        result["action"] = "scraper_called"
        return result
    except Exception as e:
        logger.exception("Error running web scraper for %s: %s", url, e)
        result["action"] = "failed"
        result["detail"]["error"] = str(e)
        return result
