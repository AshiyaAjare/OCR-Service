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

def extract_subject_company_name(text: str) -> Optional[str]:
    """
    Try to extract the *subject* company name from an equity research report.

    Heuristics:
    1. Look near the header line containing things like:
       "Equity Research", "Company Update", "Results Review", etc.
       Typically the company name is on one of the next lines, followed by a sector line
       like "Metals & Mining", "Banks", "IT Services", etc.

    2. If that fails, look above the "Market Data" block – the company name is usually
       5–10 lines before that, as a short title-case phrase.
    """
    if not text:
        return None

    # Limit to first ~2–3k chars so we stay on page 1 (where the header lives)
    header_text = text[:3000]
    lines = [ln.strip() for ln in header_text.split("\n")]

    # Common header markers in research reports
    research_header_patterns = [
        r"Equity\s+Research",
        r"Company\s+Update",
        r"Results\s+Review",
        r"Initiating\s+Coverage",
        r"Investment\s+Research",
    ]

    sector_keywords = (
        r"\b("
        r"Metals|Mining|Banks|Banking|IT|Technology|Tech|Services|Software|Pharma|Healthcare|Health\s*Care|"
        r"Auto|Automotive|FMCG|Consumer|Energy|Oil|Gas|Power|Infrastructure|Infra|Real\s+Estate|Cement|"
        r"Telecom|Media|Retail|Chemicals|Industrial|Industrials"
        r")\b"
    )

    def _looks_like_company_name(candidate: str) -> bool:
        """Basic shape + filters so we don't pick broker names, dates, or junk."""
        if not candidate:
            return False

        # Reject very short or very long
        if len(candidate) < 2 or len(candidate) > 60:
            return False

        # Reject lines with obvious non-name noise
        if any(sym in candidate for sym in ["@", "|", "www.", "http", "mailto:", "Target Price", "CMP:"]):
            return False

        # Word-count heuristic
        words = candidate.split()
        if not (1 <= len(words) <= 5):
            return False

        # Not purely numeric / date-like
        if re.search(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", candidate):
            return False

        # Avoid brokers / generic financial entities
        if re.search(r"\b(Securities|Wealth|Capital|Markets|Financial|Services)\b", candidate, re.IGNORECASE):
            return False

        # Shape: title-case or all caps is usually fine for a name like "JSW Steel"
        if candidate.isupper():
            # Avoid single all-caps words like "HOLD", "BUY"
            if len(words) == 1 and len(words[0]) <= 4:
                return False
            return True

        # First char upper; not all lower-case
        if candidate[0].isupper() and not candidate.islower():
            return True

        return False

    # --- 1) Look after research header line: "India | Equity Research | Company Update" ---
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(re.search(pat, line_lower, re.IGNORECASE) for pat in research_header_patterns):
            # Look at the next few non-empty lines for a plausible company name
            for j in range(i + 1, min(i + 7, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue

                # Skip obvious heading labels
                if re.search(r"^(rating|target|price|date|sector|industry|market\s+data)\b",
                             candidate, re.IGNORECASE):
                    continue

                if not _looks_like_company_name(candidate):
                    continue

                # Bonus: if the next line looks like a sector line, that boosts confidence
                if j + 1 < len(lines):
                    next_line = lines[j + 1].strip()
                    if re.search(sector_keywords, next_line, re.IGNORECASE):
                        return candidate

                # Even without sector confirmation, if it looks like a solid company name, accept it
                return candidate

    # --- 2) Fallback: look above "Market Data" anchor ---
    market_match = re.search(r"Market\s+Data", header_text, re.IGNORECASE)
    if market_match:
        start_pos = market_match.start()
        before_text = header_text[max(0, start_pos - 600):start_pos]
        before_lines = [ln.strip() for ln in before_text.split("\n") if ln.strip()]

        # Scan the last ~10 lines before "Market Data" for a name-like line
        for line in reversed(before_lines[-10:]):
            if _looks_like_company_name(line):
                return line

    return None


def extract_company_name(text: str) -> Optional[str]:
    """
    Extract subject company name from the document.

    Priority:
    1. Use layout-based subject-company heuristic (Equity Research / Company Update header, Market Data block).
    2. Fall back to generic '... Ltd/Limited/Inc' style patterns.
    3. Avoid returning obvious broker names like 'ICICI Securities Limited' unless nothing else is found.
    """
    if not text:
        return None

    # --- 1) Prefer a subject-company style extraction (header-based) ---
    # This uses the function you already have defined below in this file.
    try:
        subject_candidate = extract_subject_company_name(text)
    except NameError:
        # If for some reason it's not defined yet, just ignore and continue.
        subject_candidate = None

    if subject_candidate:
        return subject_candidate  # e.g. "JSW Steel"

    # --- 2) Fallback: generic pattern for legal-entity-style names ---
    patterns = [
        # Names ending with Ltd / Limited / Inc / etc.
        r'\b([A-Z][A-Za-z\s&]{3,50}?\s+(?:Ltd|Limited|Inc|Incorporated|Corporation|Corp|Private|Pvt|Public|Ltd\.))\b',
        # Names ending with common business words but without suffix
        r'\b([A-Z][A-Za-z\s&]{3,50}?\s+(?:Engineering|Technologies|Solutions|Industries|Group|Holdings))\b',
    ]

    generic_candidates: list[str] = []

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            company = match.group(1).strip()
            if not (5 <= len(company) <= 100):
                continue

            # Heuristic: skip obvious broker-style names here (Securities + Limited)
            # Example: "ICICI Securities Limited"
            if re.search(r'\bSecurities\b', company) and re.search(r'\b(Ltd|Limited)\b', company):
                continue

            generic_candidates.append(company)

        if generic_candidates:
            break

    if generic_candidates:
        # Return the first reasonable candidate we found
        return generic_candidates[0]

    # --- 3) As a last resort, allow a broker-like match if absolutely nothing else exists ---
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            company = match.group(1).strip()
            if 5 <= len(company) <= 100:
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


def extract_broker_name_from_text(text: str) -> Optional[str]:
    """
    Extract broker name from document text by looking for research report headers/footers.
    Looks for patterns like:
    - "Avendus Spark Research"
    - "ICICI Securities Research"
    - "Equity Research" headers with broker names nearby
    """
    if not text:
        return None
    
    # Limit search to first ~5000 chars (header area) and last ~3000 chars (footer area)
    header_text = text[:5000]
    footer_text = text[-3000:] if len(text) > 3000 else text
    
    # Common broker patterns (expanded list)
    broker_patterns = [
        r'\b(Avendus\s+Spark(?:\s+Research)?)\b',
        r'\b(ICICI\s+Securities?(?:\s+Research)?)\b',
        r'\b(HDFC\s+Securities?(?:\s+Research)?)\b',
        r'\b(Kotak\s+Securities?(?:\s+Research)?)\b',
        r'\b(Axis\s+Securities?(?:\s+Research)?)\b',
        r'\b(Motilal\s+Oswal(?:\s+Research)?)\b',
        r'\b(Edelweiss(?:\s+Research)?)\b',
        r'\b(Jefferies(?:\s+Research)?)\b',
        r'\b(Goldman\s+Sachs(?:\s+Research)?)\b',
        r'\b(Morgan\s+Stanley(?:\s+Research)?)\b',
        r'\b(Credit\s+Suisse(?:\s+Research)?)\b',
        r'\b(UBS(?:\s+Research)?)\b',
        r'\b(CLSA(?:\s+Research)?)\b',
        r'\b(Nomura(?:\s+Research)?)\b',
        r'\b(Macquarie(?:\s+Research)?)\b',
        r'\b(Citi(?:\s+Research)?)\b',
        r'\b(Bank\s+of\s+America(?:\s+Research)?)\b',
        # Generic pattern for "[Name] Research" or "[Name] Securities"
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:Research|Securities?))\b',
    ]
    
    # Search in header first (more likely to have broker name)
    for pattern in broker_patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            broker_name = match.group(1).strip()
            # Filter out common false positives
            if broker_name and len(broker_name) > 3:
                # Skip if it looks like a company name, not a broker
                if not re.search(r'\b(Ltd|Limited|Inc|Corporation|Corp)\b', broker_name, re.IGNORECASE):
                    return broker_name
    
    # Search in footer as fallback
    for pattern in broker_patterns:
        match = re.search(pattern, footer_text, re.IGNORECASE)
        if match:
            broker_name = match.group(1).strip()
            if broker_name and len(broker_name) > 3:
                if not re.search(r'\b(Ltd|Limited|Inc|Corporation|Corp)\b', broker_name, re.IGNORECASE):
                    return broker_name
    
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
    # Prioritize patterns that explicitly mention "Target Price" to avoid matching CMP
    target_price_patterns = [
        # Pattern 1: Table format - "CMP Target Price Rating" header, then "Rs. 1,087 Rs. 1,042 REDUCE"
        # Find "Target Price" in header, then get the second number in the next line
        r'\b(?:CMP|Current\s+Market\s+Price)\s+Target\s+Price[^\n]*\n[^\n]*(?:Rs\.?|INR|₹)?\s*[\d,]+\s+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        # Pattern 2: "Target Price: Rs. 1,042" or "Target Price Rs. 1,042" (same line)
        r'(?:Target\s+Price|Price\s+Target|TP)[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        # Pattern 3: Multi-line - "Target Price" on one line, number on next line
        # Look for "Target Price" followed by newline and then number
        r'(?:Target\s+Price|Price\s+Target|TP)[:\s]*\n\s*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
        # Pattern 4: "Rs. 1,042 Target Price" or "Rs. 1,042 Target"
        r'(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)\s*(?:Target\s+Price|Price\s+Target|Target|TP)\b',
        # Pattern 5: "Target: Rs. 1,042" (less specific, use as fallback)
        r'(?:Target)[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)',
    ]
    
    value = None
    currency = None
    rating = None
    broker_name = None
    horizon = None
    as_of_date = None
    
    # First, try to find the target price value
    # Use the most specific patterns first to avoid matching CMP
    for pattern in target_price_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        for match in matches:
            value_str = match.group(1).replace(',', '')
            try:
                candidate_value = float(value_str)
                if candidate_value > 0 and candidate_value < 1e15:
                    # Additional check: if pattern contains "CMP" nearby, skip it
                    match_start = max(0, match.start() - 20)
                    match_end = min(len(text), match.end() + 20)
                    context = text[match_start:match_end]
                    # Skip if "CMP" appears before the target price in the same context
                    if re.search(r'\bCMP\b', context[:match.start() - match_start], re.IGNORECASE):
                        # Check if there's a number after CMP that might be the CMP value
                        cmp_match = re.search(r'\bCMP[:\s]+(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d*)', context, re.IGNORECASE)
                        if cmp_match and abs(float(cmp_match.group(1).replace(',', '')) - candidate_value) < 10:
                            # This might be CMP, not target price, skip it
                            continue
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
                
                # Extract broker name (common Indian and international brokers)
                broker_patterns = [
                    r'\b(Avendus\s+Spark(?:\s+Research)?|ICICI\s+Securities?|HDFC\s+Securities?|Kotak\s+Securities?|Axis\s+Securities?|'
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
        # If broker_name not found in context, try broader extraction from document
        elif not broker_name:
            broker_name = extract_broker_name_from_text(text)
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

