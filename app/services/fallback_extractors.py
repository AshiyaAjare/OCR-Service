"""
Fallback deterministic extractors for key metadata fields.
Used when LLM extraction fails or returns invalid JSON.
"""
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from urllib.parse import urlparse


def extract_ticker(text: str) -> Optional[str]:
    """
    Extract ticker symbol from text using patterns.
    Looks for common ticker formats like "STEELCAS", "NSE:STEELCAS", etc.
    """
    if not text:
        return None
    
    # Common patterns for tickers
    patterns = [
        r'\b([A-Z]{2,10})\s+(?:Ltd|Limited|Inc|Corporation|Corp)',
        r'(?:NSE|BSE|NYSE|NASDAQ):\s*([A-Z]{2,10})\b',
        r'\b([A-Z]{2,10})\s+\(([A-Z]{2,10})\)',  # Company Name (TICKER)
        r'Ticker[:\s]+([A-Z]{2,10})\b',
        r'Symbol[:\s]+([A-Z]{2,10})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ticker = match.group(1) if match.lastindex >= 1 else match.group(0)
            # Filter out common false positives
            if ticker and len(ticker) >= 2 and len(ticker) <= 10:
                if ticker.upper() not in ['LTD', 'INC', 'CORP', 'LIMITED', 'COMPANY']:
                    return ticker.upper()
    
    return None


def extract_company_name(text: str) -> Optional[str]:
    """
    Extract company name from text.
    Looks for patterns like "Company Name Ltd", "Company Name Limited", etc.
    """
    if not text:
        return None
    
    # Pattern for company names ending with Ltd/Limited/Inc/etc
    patterns = [
        r'\b([A-Z][A-Za-z\s&]{3,50}?\s+(?:Ltd|Limited|Inc|Incorporated|Corporation|Corp|Private|Pvt|Public|Ltd\.))\b',
        r'\b([A-Z][A-Za-z\s&]{3,50}?\s+(?:Engineering|Technologies|Solutions|Industries|Group|Holdings))\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            company = match.group(1).strip()
            # Filter out very short or common words
            if len(company) >= 5 and len(company) <= 100:
                return company
    
    return None


def extract_dates(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract dates from text and normalize to ISO format.
    Returns list of dicts with 'raw' and 'normalized' keys.
    """
    if not text:
        return []
    
    dates_found = []
    
    # ISO format dates (YYYY-MM-DD)
    iso_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    for match in re.finditer(iso_pattern, text):
        dates_found.append({
            'raw': match.group(1),
            'normalized': match.group(1)
        })
    
    # Common date formats
    date_patterns = [
        (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', '%d-%m-%Y'),  # DD-MM-YYYY
        (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', '%m-%d-%Y'),  # MM-DD-YYYY
        (r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', '%d %B %Y'),
        (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', '%B %d, %Y'),
    ]
    
    for pattern, date_format in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw_date = match.group(1)
            try:
                parsed = datetime.strptime(raw_date, date_format)
                normalized = parsed.strftime('%Y-%m-%d')
                dates_found.append({
                    'raw': raw_date,
                    'normalized': normalized
                })
            except ValueError:
                dates_found.append({
                    'raw': raw_date,
                    'normalized': None
                })
    
    return dates_found


def extract_report_date(text: str) -> Optional[str]:
    """
    Extract the most likely report date from text.
    Returns ISO format string (YYYY-MM-DD) or None.
    """
    dates = extract_dates(text)
    if not dates:
        return None
    
    # Prefer normalized dates
    for date_info in dates:
        if date_info.get('normalized'):
            return date_info['normalized']
    
    return None


def extract_period(text: str) -> Optional[str]:
    """
    Extract reporting period (e.g., "Q2 2025", "H1 2025", "FY 2025").
    """
    if not text:
        return None
    
    patterns = [
        r'\b(Q[1-4]\s+\d{4})\b',  # Q1 2025, Q2 2025, etc.
        r'\b(H[12]\s+\d{4})\b',   # H1 2025, H2 2025
        r'\b(FY\s+\d{4})\b',      # FY 2025
        r'\b(Financial Year\s+\d{4})\b',
        r'\b(Year\s+Ended\s+\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_document_type(text: str) -> Optional[str]:
    """
    Extract document type from text.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    type_mapping = {
        'press release': 'press_release',
        'press_release': 'press_release',
        'quarterly results': 'results',
        'q[1-4] results': 'results',
        'annual report': 'annual_report',
        'board meeting': 'board_outcome',
        'board outcome': 'board_outcome',
        'financial results': 'results',
        'earnings': 'results',
    }
    
    for keyword, doc_type in type_mapping.items():
        if re.search(keyword, text_lower):
            return doc_type
    
    return None


def extract_detected_tickers(text: str) -> List[Dict[str, Any]]:
    """
    Extract all potential tickers from text with confidence scores.
    """
    if not text:
        return []
    
    tickers = []
    ticker = extract_ticker(text)
    
    if ticker:
        tickers.append({
            'ticker': ticker,
            'confidence': 0.7  # Medium confidence for regex-based extraction
        })
    
    return tickers


def apply_fallback_extractors(text: str, source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply all fallback extractors to text and return structured metadata.
    
    Args:
        text: The extracted text to analyze
        source_url: Optional source URL for domain extraction
    
    Returns:
        Dictionary with extracted metadata fields
    """
    result: Dict[str, Any] = {
        'ticker': extract_ticker(text),
        'company_name': extract_company_name(text),
        'report_date': extract_report_date(text),
        'period': extract_period(text),
        'document_type': extract_document_type(text),
        'raw_dates_found': extract_dates(text),
        'detected_tickers': extract_detected_tickers(text),
        'tables': [],
        'kv_pairs': [],
        'headings': [],
    }
    
    # Extract source domain from URL if provided
    if source_url:
        try:
            parsed = urlparse(source_url)
            result['source_domain'] = parsed.netloc or None
        except Exception:
            result['source_domain'] = None
    else:
        result['source_domain'] = None
    
    # Set defaults
    result['language'] = None
    result['ocr_confidence'] = None
    result['ingestion_method'] = None
    
    return result

