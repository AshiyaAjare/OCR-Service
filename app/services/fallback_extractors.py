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


def extract_broker_estimate(text: str) -> Optional[float]:
    """
    Extract broker estimate (earnings/revenue estimate) from text.
    Looks for patterns like:
    - "broker estimate: 1234.56"
    - "analyst estimate: 1234.56"
    - "consensus estimate: 1234.56"
    - "estimated: 1234.56"
    - Numbers near keywords like "estimate", "forecast", "projection"
    """
    if not text:
        return None
    
    # Common patterns for broker/analyst estimates
    patterns = [
        # Direct patterns with colons or equals
        r'(?:broker|analyst|consensus|market)\s+estimate[s]?[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        r'estimate[s]?[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        r'estimated[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        r'forecast[s]?[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        r'projection[s]?[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        # Patterns with numbers before estimate keywords
        r'(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)\s+(?:crore|lakh|million|billion)?\s*(?:broker|analyst|consensus)?\s+estimate',
        # Patterns in tables or structured formats
        r'estimate[:\s]*\n?\s*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            value_str = match.group(1).replace(',', '')  # Remove commas
            try:
                value = float(value_str)
                # Filter out obviously wrong values (too small or too large)
                # Assuming estimates are typically in reasonable ranges (e.g., 0.01 to 1e15)
                if value > 0 and value < 1e15:
                    return value
            except (ValueError, TypeError):
                continue
    
    # Try to find numbers near estimate-related keywords (context-based)
    estimate_keywords = [
        r'broker\s+estimate',
        r'analyst\s+estimate',
        r'consensus\s+estimate',
        r'estimated\s+(?:revenue|earnings|profit|sales)',
        r'forecast\s+(?:revenue|earnings|profit|sales)',
    ]
    
    for keyword_pattern in estimate_keywords:
        # Find the keyword
        keyword_matches = re.finditer(keyword_pattern, text, re.IGNORECASE)
        for keyword_match in keyword_matches:
            # Look for numbers in the next 100 characters after the keyword
            start_pos = keyword_match.end()
            context = text[start_pos:start_pos + 100]
            # Look for number patterns
            number_pattern = r'(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)\s*(?:crore|lakh|million|billion)?'
            number_matches = re.finditer(number_pattern, context, re.IGNORECASE)
            for num_match in number_matches:
                value_str = num_match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    if value > 0 and value < 1e15:
                        return value
                except (ValueError, TypeError):
                    continue
    
    return None


def extract_broker_estimate_detail(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract richer broker estimate details from text.
    Looks for patterns like:
    - "Target Price: INR 1,110 ... HOLD"
    - "Target: Rs. 1,110 (ICICI Securities) BUY"
    - "Price Target: ₹1,110 ... Rating: HOLD ... Horizon: 12M"
    
    Returns a dict with: value, estimate_type, currency, as_of_date, broker_name, rating, horizon
    """
    if not text:
        return None
    
    # Pattern to find target price with surrounding context
    # Look for "Target Price", "Target", "Price Target" followed by currency and number
    target_price_patterns = [
        r'(?:Target\s+Price|Price\s+Target|Target)[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        r'(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)\s*(?:Target|Target\s+Price)',
    ]
    
    value = None
    currency = None
    rating = None
    broker_name = None
    horizon = None
    as_of_date = None
    
    # First, try to find the target price value
    for pattern in target_price_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        for match in matches:
            value_str = match.group(1).replace(',', '')
            try:
                candidate_value = float(value_str)
                if candidate_value > 0 and candidate_value < 1e15:
                    value = candidate_value
                    # Extract currency from the match
                    match_text = match.group(0)
                    if 'INR' in match_text.upper() or '₹' in match_text or 'Rs' in match_text.upper():
                        currency = 'INR'
                    break
            except (ValueError, TypeError):
                continue
        if value:
            break
    
    # If we found a value, look for additional context in nearby text
    if value:
        # Find the position where we found the target price
        for pattern in target_price_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Look in a 200-character window around the match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 150)
                context = text[start:end]
                
                # Extract rating (HOLD, BUY, SELL, etc.)
                rating_patterns = [
                    r'\b(Rating|Rating:)\s*(HOLD|BUY|SELL|STRONG\s+BUY|STRONG\s+SELL|NEUTRAL|ACCUMULATE|REDUCE)\b',
                    r'\b(HOLD|BUY|SELL|STRONG\s+BUY|STRONG\s+SELL|NEUTRAL|ACCUMULATE|REDUCE)\s*(?:Rating|\(Rating\))?\b',
                ]
                for rating_pattern in rating_patterns:
                    rating_match = re.search(rating_pattern, context, re.IGNORECASE)
                    if rating_match:
                        rating = rating_match.group(2) if rating_match.lastindex >= 2 else rating_match.group(1)
                        rating = rating.upper().strip()
                        break
                
                # Extract broker name (common Indian brokers)
                broker_patterns = [
                    r'\b(ICICI\s+Securities?|HDFC\s+Securities?|Kotak\s+Securities?|Axis\s+Securities?|'
                    r'Motilal\s+Oswal|Edelweiss|Jefferies|Goldman\s+Sachs|Morgan\s+Stanley|'
                    r'Credit\s+Suisse|UBS|CLSA|Nomura|Macquarie|Citi|Bank\s+of\s+America)\b',
                    r'\b([A-Z][a-z]+\s+Securities?)\b',
                ]
                for broker_pattern in broker_patterns:
                    broker_match = re.search(broker_pattern, context, re.IGNORECASE)
                    if broker_match:
                        broker_name = broker_match.group(1).strip()
                        break
                
                # Extract horizon (12M, 6M, etc.)
                horizon_patterns = [
                    r'\b(Horizon|Timeframe|Period)[:\s]+([^,\n]+?)(?:\.|,|\n|$)',
                    r'\b(\d+\s*[Mm]onths?|\d+\s*[Yy]ears?|12M|6M|18M|24M)\b',
                ]
                for horizon_pattern in horizon_patterns:
                    horizon_match = re.search(horizon_pattern, context, re.IGNORECASE)
                    if horizon_match:
                        horizon = horizon_match.group(2) if horizon_match.lastindex >= 2 else horizon_match.group(1)
                        horizon = horizon.strip()
                        break
                
                # Extract date near the target price
                date_pattern = r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'
                date_match = re.search(date_pattern, context)
                if date_match:
                    date_str = date_match.group(1)
                    # Try to normalize to YYYY-MM-DD
                    try:
                        if '-' in date_str and len(date_str.split('-')) == 3:
                            parts = date_str.split('-')
                            if len(parts[0]) == 4:  # YYYY-MM-DD
                                as_of_date = date_str
                            else:  # DD-MM-YYYY
                                d, m, y = parts
                                as_of_date = f"{y}-{m}-{d}"
                        elif '/' in date_str:
                            parts = date_str.split('/')
                            if len(parts) == 3:
                                if len(parts[2]) == 4:  # DD/MM/YYYY or MM/DD/YYYY
                                    # Try DD/MM/YYYY first (common in India)
                                    d, m, y = parts
                                    as_of_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    except Exception:
                        pass
                
                break
    
    # If we found a value, return the detail structure
    if value:
        detail = {
            "value": value,
            "estimate_type": "target_price",
            "currency": currency or "INR",  # Default to INR for Indian context
        }
        if as_of_date:
            detail["as_of_date"] = as_of_date
        if broker_name:
            detail["broker_name"] = broker_name
        if rating:
            detail["rating"] = rating
        if horizon:
            detail["horizon"] = horizon
        
        return detail
    
    return None


def apply_fallback_extractors(text: str, source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply all fallback extractors to text and return structured metadata.
    
    Args:
        text: The extracted text to analyze
        source_url: Optional source URL for domain extraction
    
    Returns:
        Dictionary with extracted metadata fields
    """
    # Extract broker estimate detail (richer structure)
    broker_estimate_detail = extract_broker_estimate_detail(text)
    
    # Extract broker_estimate value (numeric) - prefer from detail, fallback to simple extractor
    broker_estimate_value = None
    if broker_estimate_detail and isinstance(broker_estimate_detail, dict):
        broker_estimate_value = broker_estimate_detail.get("value")
    else:
        broker_estimate_value = extract_broker_estimate(text)
    
    result: Dict[str, Any] = {
        'ticker': extract_ticker(text),
        'company_name': extract_company_name(text),
        'report_date': extract_report_date(text),
        'period': extract_period(text),
        'document_type': extract_document_type(text),
        'raw_dates_found': extract_dates(text),
        'detected_tickers': extract_detected_tickers(text),
        'broker_estimate': broker_estimate_value,
        'broker_estimate_detail': broker_estimate_detail,
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

